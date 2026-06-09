require("dotenv").config();

const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");

const express = require("express");

const { GoogleGenerativeAI } = require("@google/generative-ai");

const {
  RecursiveCharacterTextSplitter,
} = require("@langchain/textsplitters");

const { QdrantClient } = require("@qdrant/js-client-rest");

// ======================================
// GEMINI
// ======================================

const genAI = new GoogleGenerativeAI(
  process.env.GEMINI_API_KEY
);

// Embedding Model
const embeddingModel =
  genAI.getGenerativeModel({
    model: "gemini-embedding-2",
  });

// Chat Model
const chatModel =
  genAI.getGenerativeModel({
    model: "gemini-3-flash-preview",
  });

// ======================================
// QDRANT
// ======================================

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

const COLLECTION_NAME = "insurance_rag";

// ======================================
// EMBEDDING FUNCTION
// ======================================

async function createEmbedding(
  text,
  retries = 5
) {
  for (
    let attempt = 1;
    attempt <= retries;
    attempt++
  ) {
    try {
      const result =
        await embeddingModel.embedContent({
          content: {
            parts: [{ text }],
          },
        });

      return result.embedding.values;
    } catch (err) {
      if (err.status === 429) {
        console.log(
          `Rate limit. Retry ${attempt}`
        );

        await new Promise((resolve) =>
          setTimeout(resolve, 5000)
        );
      } else {
        throw err;
      }
    }
  }
}

// ======================================
// INIT VECTOR DB
// ======================================

async function initDB() {
  try {
    await qdrant.createCollection(
      COLLECTION_NAME,
      {
        vectors: {
          size: 3072,
          distance: "Cosine",
        },
      }
    );

    console.log(
      "Qdrant Collection Created"
    );
  } catch {
    console.log(
      "Collection Already Exists"
    );
  }
}

// ======================================
// LOAD PDFs
// ======================================

async function loadPDFs(folderPath) {
  const documents = [];

  const files =
    fs.readdirSync(folderPath);

  for (const file of files) {
    if (
      path.extname(file).toLowerCase() !==
      ".pdf"
    )
      continue;

    console.log(`Reading ${file}`);

    const fullPath = path.join(
      folderPath,
      file
    );

    const buffer =
      fs.readFileSync(fullPath);

    // console.log(pdfParse.PDFParse);

    const data =
      await new pdfParse.PDFParse(Uint8Array.from(buffer)).getText();
    // console.log(` ${data.text}`);
    documents.push({
      fileName: file,
      text: data.text,
    });
  }

  return documents;
}

// ======================================
// CHUNKING
// ======================================

async function createChunks(
  documents
) {
  const splitter =
    new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

  const finalChunks = [];

  for (const doc of documents) {
    const chunks =
      await splitter.splitText(
        doc.text
      );

    chunks.forEach(
      (chunk, index) => {
        finalChunks.push({
          id: `${doc.fileName}-${index}`,
          text: chunk,
          metadata: {
            source: doc.fileName,
          },
        });
      }
    );
  }

  return finalChunks;
}

// ======================================
// BUILD VECTOR DB
// ======================================

async function buildVectorDB(
  chunks
) {
  console.log(
    "\nCreating embeddings...\n"
  );

  const existing =
    await qdrant.getCollection(
      COLLECTION_NAME
    );

  if (
    existing.points_count &&
    existing.points_count > 0
  ) {
    console.log(
      "Embeddings already exist"
    );

    return;
  }

  const BATCH_SIZE = 2;

  for (
    let i = 0;
    i < chunks.length;
    i += BATCH_SIZE
  ) {
    const batch = chunks.slice(
      i,
      i + BATCH_SIZE
    );

    console.log(
      `Batch ${i / BATCH_SIZE + 1}`
    );

    const points =
      await Promise.all(
        batch.map(
          async (chunk, index) => {
            const embedding =
              await createEmbedding(
                chunk.text
              );

            return {
              id: i + index,

              vector: embedding,

              payload: {
                text: chunk.text,
                source:
                  chunk.metadata.source,
              },
            };
          }
        )
      );

    await qdrant.upsert(
      COLLECTION_NAME,
      {
        wait: true,
        points,
      }
    );

    await new Promise((resolve) =>
      setTimeout(resolve, 1500)
    );
  }

  console.log("\nVector DB Ready");
}

// ======================================
// RETRIEVAL
// ======================================

async function retrieve(
  query,
  topK = 5
) {
  const queryEmbedding =
    await createEmbedding("all plans with plan name, coverage, term period and returns");

  const results =
    await qdrant.search(
      COLLECTION_NAME,
      {
        vector: queryEmbedding,
        limit: topK,
      }
    );

  return results.map((item) => ({
    text: item.payload.text,
    source: item.payload.source,
    score: item.score,
  }));
}

// ======================================
// GENERATE RESPONSE
// ======================================

async function askQuestion(query) {
  const retrievedChunks =
    await retrieve(query);

  const context =
    retrievedChunks
      .map((item) => item.text)
      .join("\n\n");

  const prompt = `
          You are an intelligent insurance advisor.

          Answer ONLY using the provided context.

          Context:
          ${context}

          User Question:
          ${query}
          `;
  console.log(prompt);

  const result =
    await chatModel.generateContent(
      prompt
    );


  return result.response.text();
}

// ======================================
// INITIALIZE SYSTEM
// ======================================

async function initializeRAG() {
  await initDB();

  const documents =
    await loadPDFs("./PDFs");

  const chunks =
    await createChunks(documents);

  console.log(
    `Total Chunks: ${chunks.length}`
  );

  await buildVectorDB(chunks);
}

// ======================================
// EXPRESS SERVER
// ======================================

const app = express();

app.use(express.json());

// ======================================
// SINGLE API
// ======================================

app.get("/", (req, res) => {
  res.send(
    "RAG AI Module is Running"
  );
});

app.get("/ask", async (req, res) => {
  try {
    const question = req.query.question;

    if (!question) {
      return res.json({
        success: false,
        message:
          "Pass question in query params",
      });
    }

    const answer =
      await askQuestion(question);

    res.json({
      success: true,
      question,
      answer,
    });
  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message,
    });
  }
});

app.post("/ask", async (req, res) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({
        success: false,
        message:
          "Question is required",
      });
    }

    const answer =
      await askQuestion(question);

    res.json({
      success: true,
      question,
      answer,
    });
  } catch (err) {
    console.error(err);

    res.status(500).json({
      success: false,
      error: err.message,
    });
  }
});

// ======================================
// START SERVER
// ======================================

(async () => {
  await initializeRAG();

  const PORT = 3001;

  app.listen(PORT, "0.0.0.0", () => {
    console.log(
      `Server Running On Port ${PORT}`
    );

    console.log(
      `URL: https://${process.env.CODESPACE_NAME}-${PORT}.app.github.dev`
    );
  });
})();