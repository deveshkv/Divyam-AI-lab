import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import fs from 'fs';
import { createRequire } from 'module';

// Logging helper
const LOG_FILE = path.join(process.cwd(), 'error.log');

function logToFile(level: string, message: string, error?: any) {
  const timestamp = new Date().toISOString();
  const logEntry = `[${timestamp}] [${level}] ${message}${error ? `\nError: ${error.stack || error}` : ''}\n---\n`;
  try {
    fs.appendFileSync(LOG_FILE, logEntry);
  } catch (e) {
    console.error('Failed to write to log file:', e);
  }
}

import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

const require = createRequire(import.meta.url);
const pdfImport = require('pdf-parse');
const mammothImport = require('mammoth');
const multerImport = require('multer');

// Robust module interop
const pdf = pdfImport.PDFParse || pdfImport.default || pdfImport;
const mammoth = mammothImport.default || mammothImport;
const multer = multerImport.default || multerImport;

const { parse: parseCsv } = require('csv-parse/sync');

dotenv.config();

// Debug logs for imports
try {
  logToFile('DEBUG', `Imports check - pdf: ${typeof pdf}, mammoth: ${typeof mammoth}, multer: ${typeof multer}`);
  logToFile('DEBUG', `pdf keys: ${Object.keys(pdf || {}).join(', ')}`);
  logToFile('DEBUG', `mammoth keys: ${Object.keys(mammoth || {}).join(', ')}`);
} catch (e) {}

if (!process.env.GEMINI_API_KEY && !process.env.GEMINI_API_KEY1) {
  console.warn('⚠️  WARNING: GEMINI_API_KEY or GEMINI_API_KEY1 is not set in environment variables.');
}

const app = express();
const PORT = 3000;

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 } // 20MB limit
});

// In-memory cache for queries
const queryCache = new Map<string, {
  answer: string;
  sources: any[];
  metrics: any;
}>();

// In-memory vector store for MVP
interface VectorEntry {
  embedding: number[];
  text: string;
  metadata: any;
}

let vectorIndex: VectorEntry[] = [];
let currentFileStats = {
  chunks: 0,
  embeddings: 0,
  processingTime: 0,
};

// Progress tracking for ingestion
let ingestionProgress = {
  step: 0,
  totalSteps: 4,
  currentStepName: 'idle',
  completedSteps: [] as string[],
  error: null as string | null
};

let aiClient: GoogleGenAI | null = null;

function getAiClient(): GoogleGenAI {
  if (!aiClient) {
    let apiKey = process.env.GEMINI_API_KEY;
    
    // Check for GEMINI_API_KEY1 if GEMINI_API_KEY is missing or a placeholder
    if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
      const altKey = process.env.GEMINI_API_KEY1;
      if (altKey && altKey !== 'MY_GEMINI_API_KEY') {
        logToFile('INFO', '[AI Client] Using GEMINI_API_KEY1 as fallback');
        apiKey = altKey;
      }
    }

    if (!apiKey) {
      logToFile('ERROR', '[AI Client] GEMINI_API_KEY is missing');
      throw new Error('GEMINI_API_KEY is not configured. Please add your real Gemini API key to the AI Studio Secrets panel.');
    }
    
    if (apiKey === 'MY_GEMINI_API_KEY') {
      logToFile('WARN', '[AI Client] GEMINI_API_KEY is the default placeholder. This might fail unless AI Studio injects the real key at runtime.');
    }
    
    logToFile('DEBUG', `[AI Client] Initializing with API key (length: ${apiKey.length}, starts with: ${apiKey.slice(0, 4)}...)`);
    aiClient = new GoogleGenAI({ apiKey });
  }
  return aiClient;
}

// Helper: Cosine Similarity
function cosineSimilarity(a: number[], b: number[]) {
  let dotProduct = 0;
  let mA = 0;
  let mB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    mA += a[i] * a[i];
    mB += b[i] * b[i];
  }
  return dotProduct / (Math.sqrt(mA) * Math.sqrt(mB));
}

/**
 * Advanced chunking strategy:
 * - Splits text into chunks of a target size.
 * - Includes an overlap between consecutive chunks to maintain context.
 * - Attempts to break at sentence boundaries (., !, ?) or paragraph breaks (\n) 
 *   to avoid cutting mid-sentence, provided the break is within a reasonable range.
 */
function chunkText(text: string, chunkSize: number = 1000, overlap: number = 200): string[] {
  if (!text) return [];
  
  const result: string[] = [];
  let currentPos = 0;
  
  while (currentPos < text.length) {
    let endPos = currentPos + chunkSize;
    
    if (endPos < text.length) {
      // Look for sentence or paragraph breaks near the end of the chunk
      const lastPeriod = text.lastIndexOf('. ', endPos);
      const lastExclamation = text.lastIndexOf('! ', endPos);
      const lastQuestion = text.lastIndexOf('? ', endPos);
      const lastNewline = text.lastIndexOf('\n', endPos);
      
      const breakPoint = Math.max(lastPeriod, lastExclamation, lastQuestion, lastNewline);
      
      // If a break point is found and it's within the last 40% of the chunk, use it
      if (breakPoint > currentPos + (chunkSize * 0.6)) {
        endPos = breakPoint + 1;
      }
    }
    
    const chunk = text.substring(currentPos, endPos).trim();
    if (chunk) {
      result.push(chunk);
    }
    
    // Move forward, ensuring we always make progress even if overlap is large
    const nextPos = endPos - overlap;
    if (nextPos <= currentPos) {
      currentPos = endPos; // Force progress if overlap would keep us stuck
    } else {
      currentPos = nextPos;
    }
    
    if (currentPos >= text.length) break;
  }
  
  return result;
}

async function startServer() {
  logToFile('INFO', 'startServer() starting...');
  
  const apiRouter = express.Router();
  apiRouter.use(express.json());

  // API: Progress
  apiRouter.get('/upload-progress', (req, res) => {
    logToFile('DEBUG', '[API] GET /api/upload-progress requested');
    res.json(ingestionProgress);
  });

  // API: Health Check
  apiRouter.get('/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  // API: File Upload & Process
  logToFile('DEBUG', 'Registering POST /api/upload');
  apiRouter.post('/upload', (req, res, next) => {
    upload.single('file')(req, res, (err) => {
      if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
        logToFile('ERROR', '[Upload] File too large');
        return res.status(400).json({ error: 'File is too large. Maximum size is 20MB.' });
      } else if (err) {
        return next(err);
      }
      next();
    });
  }, async (req, res) => {
    const startTime = Date.now();
    logToFile('INFO', '[Upload] Request received');
    
    const file = (req as any).file;
    logToFile('DEBUG', `[Upload] req.file: ${file ? `${file.originalname} (${file.size} bytes)` : 'undefined'}`);

    if (!process.env.GEMINI_API_KEY && !process.env.GEMINI_API_KEY1) {
      logToFile('ERROR', '[Upload] GEMINI_API_KEY is missing');
      return res.status(500).json({ error: 'GEMINI_API_KEY is not configured. Please add it to the AI Studio Secrets panel.' });
    }

    if (!file) {
      logToFile('ERROR', '[Upload] No file uploaded');
      return res.status(400).json({ error: 'No file uploaded.' });
    }

    ingestionProgress = { step: 1, totalSteps: 4, currentStepName: 'Parsing Document', completedSteps: [], error: null };
    
    try {
      const file = (req as any).file;
      if (!file) {
        logToFile('WARN', '[Upload] No file provided');
        ingestionProgress.error = 'No file uploaded';
        return res.status(400).json({ error: 'No file uploaded. Please select a valid document.' });
      }

      let text = '';
      const buffer = file.buffer;
      const filename = file.originalname;
      const ext = path.extname(filename).toLowerCase();

      logToFile('INFO', `[Upload] Processing file: ${filename} (${file.size} bytes), ext: ${ext}`);

      try {
        if (ext === '.pdf') {
          logToFile('DEBUG', `Using pdf parser of type: ${typeof pdf}`);
          try {
            let data;
            try {
              // Standard pdf-parse is a function
              data = await pdf(buffer);
            } catch (err: any) {
              if (err.message?.includes("Class constructor") || err.message?.includes("without 'new'")) {
                logToFile('DEBUG', '[Upload] Retrying PDF parse with "new"');
                data = await new (pdf as any)(buffer);
              } else {
                throw err;
              }
            }
            logToFile('DEBUG', `[Upload] PDF parse data keys: ${Object.keys(data || {})}`);
            text = data.text || '';
          } catch (err: any) {
            logToFile('ERROR', `[Upload] PDF parse failed: ${err.message}`);
            throw new Error(`Failed to parse PDF: ${err.message}`);
          }
        } else if (ext === '.csv') {
          const records = parseCsv(buffer);
          text = records.map((r: any) => Object.values(r).join(' ')).join('\n');
        } else if (ext === '.docx') {
          const result = await mammoth.extractRawText({ buffer });
          text = result.value || '';
        } else {
          text = buffer.toString('utf-8') || '';
        }
        logToFile('DEBUG', `[Upload] Extracted text length: ${text.length}`);
      } catch (parseError: any) {
        logToFile('ERROR', `[Upload] Parsing failed for ${filename}`, parseError);
        ingestionProgress.error = `Parsing failed: ${parseError.message}`;
        return res.status(422).json({ error: `Failed to parse ${ext.toUpperCase()} file. It might be corrupted or password-protected.` });
      }

      const trimmedText = (text || '').trim();
      if (!trimmedText) {
        logToFile('WARN', `[Upload] Empty text extracted from ${filename}`);
        ingestionProgress.error = 'Empty text extracted';
        return res.status(400).json({ error: 'The uploaded file appears to be empty or contains no extractable text.' });
      }

      ingestionProgress.completedSteps.push('Parsing Document');
      ingestionProgress.step = 2;
      ingestionProgress.currentStepName = 'Chunking & Tokenization';

      // Chunking with overlap and sentence awareness
      const chunks = chunkText(text, 1000, 200);
      logToFile('INFO', `[Upload] Generated ${chunks.length} chunks`);
      
      ingestionProgress.completedSteps.push('Chunking & Tokenization');
      ingestionProgress.step = 3;
      ingestionProgress.currentStepName = 'Vectorization (Embedding)';

      // Clear old index and cache
      vectorIndex = [];
      queryCache.clear();
      
      // Generate Embeddings
      const embeddingModel = 'gemini-embedding-2-preview';
      
      try {
        const batchSize = 100; // Increased from 10
        const concurrentRequests = 5;
        const batches = [];
        
        for (let i = 0; i < chunks.length; i += batchSize) {
          batches.push({
            batch: chunks.slice(i, i + batchSize),
            startIndex: i
          });
        }

        logToFile('INFO', `[Upload] Processing ${batches.length} batches with concurrency ${concurrentRequests}`);

        for (let i = 0; i < batches.length; i += concurrentRequests) {
          const currentBatches = batches.slice(i, i + concurrentRequests);
          await Promise.all(currentBatches.map(async (b) => {
            const result = await getAiClient().models.embedContent({
              model: embeddingModel,
              contents: b.batch,
            });

            result.embeddings.forEach((emb, idx) => {
              vectorIndex.push({
                embedding: emb.values,
                text: b.batch[idx],
                metadata: { source: filename, chunkIndex: b.startIndex + idx },
              });
            });
          }));
          
          // Update progress periodically
          const progressPercent = Math.min(100, Math.round(((i + currentBatches.length) / batches.length) * 100));
          ingestionProgress.currentStepName = `Vectorization (${progressPercent}%)`;
        }
      } catch (embError: any) {
        logToFile('ERROR', '[Upload] Embedding generation failed', embError);
        ingestionProgress.error = 'Embedding failed';
        
        const errorMsg = embError.message || '';
        if (errorMsg.includes('API key not valid') || errorMsg.includes('INVALID_ARGUMENT')) {
          return res.status(401).json({ error: 'The Gemini API key is invalid or missing. Please ensure it is correctly configured in the AI Studio Secrets panel.' });
        }
        
        return res.status(502).json({ error: 'Failed to generate embeddings. The AI service might be temporarily unavailable.' });
      }

      ingestionProgress.completedSteps.push('Vectorization (Embedding)');
      ingestionProgress.step = 4;
      ingestionProgress.currentStepName = 'Finalizing Index';

      currentFileStats = {
        chunks: chunks.length,
        embeddings: vectorIndex.length,
        processingTime: Date.now() - startTime,
      };

      ingestionProgress.completedSteps.push('Finalizing Index');
      ingestionProgress.step = 5; // Completed
      ingestionProgress.currentStepName = 'Ready';

      res.json({ success: true, stats: currentFileStats });
    } catch (error: any) {
      logToFile('ERROR', '[Upload] Unexpected error', error);
      ingestionProgress.error = 'Unexpected error';
      res.status(500).json({ error: 'An unexpected error occurred during file processing. Please try again.' });
    }
  });

  apiRouter.post('/clear-cache', (req, res) => {
    queryCache.clear();
    vectorIndex = [];
    ingestionProgress = { step: 0, totalSteps: 4, currentStepName: 'idle', completedSteps: [], error: null };
    res.json({ success: true });
  });

  // API: Test Pipeline with local file
  apiRouter.post('/test-pipeline', async (req, res) => {
    const startTime = Date.now();
    const testFilePath = path.join(process.cwd(), 'test_rag.txt');
    
    if (!fs.existsSync(testFilePath)) {
      return res.status(404).json({ error: 'Test file not found. Please create it first.' });
    }

    try {
      const text = fs.readFileSync(testFilePath, 'utf-8');
      logToFile('INFO', `[Test Pipeline] Processing test file: test_rag.txt (${text.length} chars)`);

      // Clear old index
      vectorIndex = [];
      queryCache.clear();

      // Chunking with overlap and sentence awareness
      const chunks = chunkText(text, 1000, 200);
      
      // Embedding
      const embeddingModel = 'gemini-embedding-2-preview';
      const batchSize = 100;
      const concurrentRequests = 5;
      const batches = [];
      
      for (let i = 0; i < chunks.length; i += batchSize) {
        batches.push({
          batch: chunks.slice(i, i + batchSize),
          startIndex: i
        });
      }

      for (let i = 0; i < batches.length; i += concurrentRequests) {
        const currentBatches = batches.slice(i, i + concurrentRequests);
        await Promise.all(currentBatches.map(async (b) => {
          const result = await getAiClient().models.embedContent({
            model: embeddingModel,
            contents: b.batch,
          });

          result.embeddings.forEach((emb, idx) => {
            vectorIndex.push({
              embedding: emb.values,
              text: b.batch[idx],
              metadata: { source: 'test_rag.txt', chunkIndex: b.startIndex + idx },
            });
          });
        }));
      }

      currentFileStats = {
        chunks: chunks.length,
        embeddings: vectorIndex.length,
        processingTime: Date.now() - startTime,
      };

      res.json({ success: true, stats: currentFileStats });
    } catch (error: any) {
      logToFile('ERROR', '[Test Pipeline] Failed', error);
      res.status(500).json({ error: error.message });
    }
  });

  // API: Query
  apiRouter.post('/query', async (req, res) => {
    const startTime = Date.now();
    const { prompt } = req.body;

    if (!prompt || typeof prompt !== 'string') {
      logToFile('WARN', '[Query] Invalid prompt received');
      return res.status(400).json({ error: 'Invalid query provided.' });
    }

    // Token limit check
    if (prompt.length > 4000) {
      logToFile('WARN', `[Query] Prompt too long: ${prompt.length} chars`);
      return res.status(400).json({ error: 'Query is too long. Please limit your question to approximately 1000 words.' });
    }

    // 0. Cache Check (Normalized + Hash-like key)
    const normalizedQuery = prompt.trim().toLowerCase().replace(/[^\w\s]/g, '').replace(/\s+/g, ' ');
    if (queryCache.has(normalizedQuery)) {
      logToFile('INFO', `[Query] Cache hit for: "${prompt}"`);
      const cached = queryCache.get(normalizedQuery)!;
      return res.json({ ...cached, cached: true });
    }

    try {
      if (vectorIndex.length === 0) {
        logToFile('WARN', '[Query] Attempted query without document');
        return res.status(400).json({ error: 'No document processed yet. Please upload a file first.' });
      }

      logToFile('INFO', `[Query] Processing: "${prompt}"`);

      // 1. Query Optimization Layer (Skipped for speed)
      const optimizedQuery = prompt;
      
      // 2. Embed Query
      let queryEmbedding;
      try {
        const queryEmbeddingResult = await getAiClient().models.embedContent({
          model: 'gemini-embedding-2-preview',
          contents: [optimizedQuery],
        });
        queryEmbedding = queryEmbeddingResult.embeddings[0].values;
      } catch (embError: any) {
        logToFile('ERROR', '[Query] Query embedding failed', embError);
        const errorMsg = embError.message || '';
        if (errorMsg.includes('API key not valid') || errorMsg.includes('INVALID_ARGUMENT')) {
          return res.status(401).json({ error: 'The Gemini API key is invalid or missing. Please check your settings.' });
        }
        return res.status(502).json({ error: 'Failed to process query embeddings.' });
      }

      // 3. Retrieve Top-K (k=5)
      const scored = vectorIndex.map(entry => ({
        ...entry,
        score: cosineSimilarity(queryEmbedding, entry.embedding)
      }));
      scored.sort((a, b) => b.score - a.score);
      const topK = scored.slice(0, 5);

      // 4. Generate Answer
      const context = topK.map(t => t.text).join('\n---\n');
      const systemInstruction = `You are a helpful AI assistant. Use the provided context to answer the user's question. 
      If the answer is not in the context, say you don't know based on the document.
      Context:
      ${context}`;

      const response = await getAiClient().models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: prompt,
        config: { systemInstruction }
      });

      const latency = Date.now() - startTime;
      const tokens = response.usageMetadata?.totalTokenCount || 0;
      const cost = (tokens / 1000000) * 0.075;

      const result = {
        answer: response.text,
        sources: topK.map(t => ({ text: t.text, metadata: t.metadata })),
        metrics: {
          latency,
          tokens,
          cost,
          retrievalCount: topK.length
        }
      };

      // Store in cache
      queryCache.set(normalizedQuery, result);

      res.json(result);
    } catch (error: any) {
      logToFile('ERROR', '[Query] Unexpected error', error);
      res.status(500).json({ error: 'An error occurred while generating the answer. Please try again.' });
    }
  });

  // API: Suggestions
  apiRouter.get('/suggestions', async (req, res) => {
    if (vectorIndex.length === 0) {
      return res.json({ suggestions: ['What is this document about?', 'Summarize the key points', 'Who are the main entities mentioned?'] });
    }

    try {
      const sampleText = vectorIndex.slice(0, 3).map(v => v.text).join('\n');
      const response = await getAiClient().models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: `Based on the following document snippets, generate 3 concise, high-value questions a user might want to ask.
        Output ONLY the questions, one per line.
        Snippets:
        ${sampleText}`,
      });

      const suggestions = response.text?.split('\n').filter(s => s.trim()).slice(0, 3) || [];
      res.json({ suggestions });
    } catch (error) {
      res.json({ suggestions: ['What is this document about?', 'Summarize the key points', 'Who are the main entities mentioned?'] });
    }
  });

  // API: Generate Test Questions
  apiRouter.get('/generate-test-questions', async (req, res) => {
    try {
      if (vectorIndex.length === 0) {
        return res.status(400).json({ error: 'No document processed yet.' });
      }

      const sampleText = vectorIndex.slice(0, 10).map(v => v.text).join('\n');
      const response = await getAiClient().models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: `Generate 5 challenging test questions based on the following document content. 
        These questions will be used to evaluate a RAG pipeline.
        Output ONLY the questions, one per line.
        Text: 
        ${sampleText}`,
      });

      const questions = response.text?.split('\n').filter(s => s.trim()).slice(0, 5) || [];
      res.json({ questions });
    } catch (error: any) {
      logToFile('ERROR', '[Eval] Failed to generate test questions', error);
      res.status(500).json({ error: 'Failed to generate test questions.' });
    }
  });

  // API: Evaluate Pipeline
  apiRouter.post('/evaluate', async (req, res) => {
    const { questions } = req.body;
    if (!questions || !Array.isArray(questions)) {
      return res.status(400).json({ error: 'Invalid questions provided.' });
    }

    if (vectorIndex.length === 0) {
      return res.status(400).json({ error: 'No document processed yet.' });
    }

    logToFile('INFO', `[Eval] Starting evaluation for ${questions.length} questions`);

    const results = [];
    for (const question of questions) {
      try {
        // 1. Retrieve
        const queryEmbeddingResult = await getAiClient().models.embedContent({
          model: 'gemini-embedding-2-preview',
          contents: [question],
        });
        const queryEmbedding = queryEmbeddingResult.embeddings[0].values;
        
        const scored = vectorIndex.map(entry => ({
          ...entry,
          score: cosineSimilarity(queryEmbedding, entry.embedding)
        }));
        scored.sort((a, b) => b.score - a.score);
        const topK = scored.slice(0, 5);
        const context = topK.map(t => t.text).join('\n---\n');

        // 2. Generate
        const systemInstruction = `Use the provided context to answer the question. 
        If the answer is not in the context, say you don't know.
        Context: ${context}`;

        const genResponse = await getAiClient().models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: question,
          config: { systemInstruction }
        });
        const answer = genResponse.text || '';

        // 3. Score Faithfulness
        const faithResponse = await getAiClient().models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: `Evaluate the faithfulness of the following answer based on the provided context.
          Faithfulness means the answer is derived ONLY from the context and does not contain hallucinations.
          Output a score between 0 and 1, where 1 is perfectly faithful and 0 is not faithful at all.
          Output ONLY the numeric score.
          
          Context: ${context}
          Answer: ${answer}`,
        });
        const faithfulness = parseFloat(faithResponse.text?.trim() || '0');

        // 4. Score Relevance
        const relResponse = await getAiClient().models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: `Evaluate the relevance of the following answer to the user's question.
          Relevance means the answer directly addresses the question asked.
          Output a score between 0 and 1, where 1 is perfectly relevant and 0 is not relevant at all.
          Output ONLY the numeric score.
          
          Question: ${question}
          Answer: ${answer}`,
        });
        const relevance = parseFloat(relResponse.text?.trim() || '0');

        results.push({
          question,
          answer,
          faithfulness,
          relevance
        });
      } catch (err) {
        logToFile('ERROR', `[Eval] Failed for question: ${question}`, err);
        results.push({
          question,
          answer: 'Error during evaluation',
          faithfulness: 0,
          relevance: 0
        });
      }
    }

    const avgFaithfulness = results.reduce((acc, r) => acc + r.faithfulness, 0) / results.length;
    const avgRelevance = results.reduce((acc, r) => acc + r.relevance, 0) / results.length;

    res.json({
      results,
      metrics: {
        avgFaithfulness,
        avgRelevance,
        totalQuestions: questions.length
      }
    });
  });

  // Catch-all for undefined API routes to prevent SPA fallback
  apiRouter.all('*', (req, res) => {
    logToFile('WARN', `[API] Route not found: ${req.method} ${req.url}`);
    res.status(404).json({ error: `API route not found: ${req.method} ${req.url}` });
  });

  // Mount API Router
  app.use('/api', apiRouter);

  logToFile('INFO', `Vite mode: ${process.env.NODE_ENV !== 'production' ? 'development' : 'production'}`);
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  logToFile('INFO', `Attempting to listen on port ${PORT}`);
  app.listen(PORT, '0.0.0.0', () => {
    logToFile('INFO', `Server running on http://localhost:${PORT}`);
    console.log('🚀 ML Studio Server starting...');
    console.log(`📡 Listening on http://localhost:${PORT}`);
    console.log(`🛠️ Environment: ${process.env.NODE_ENV || 'development'}`);
  });

  // Global Error Handler
  app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    logToFile('ERROR', '[Global] Unhandled error', err);
    res.status(500).json({ 
      error: 'An unexpected server error occurred.',
      message: err.message 
    });
  });
}

startServer().catch(err => {
  console.error('❌ Failed to start server:', err);
  logToFile('ERROR', 'Failed to start server', err);
});
