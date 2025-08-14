import { ChromaClient } from "chromadb";
import { pipeline } from "@xenova/transformers";

// Chroma-Client: v2-Server, host/port (kein "path")
const chroma = new ChromaClient({ host: "localhost", port: 8000 });

// Embedding-Modell lokal (wird beim ersten Lauf gecached)
const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

// Hilfsfunktionen: mean pooling + L2-Normalisierung
const l2 = v => Math.hypot(...v);
const norm = v => { const n = l2(v); return n ? v.map(x => x / n) : v; };
async function embedOne(text) {
  const out = await embedder(text, { pooling: "mean", normalize: false });
  return norm(Array.from(out.data));           // number[]
}

// Eigene EmbeddingFunction → vermeidet @chroma-core/default-embed
const embeddingFunction = { generate: async (texts) => Promise.all(texts.map(embedOne)) };

const collection = await chroma.getOrCreateCollection({
  name: "docs-demo",                            // nimm gerne einen neuen Namen
  embeddingFunction
});

// Testdaten (du kannst hier alles reinkippen)
const docs = [
  "Unser Rückgaberecht beträgt 30 Tage.",
  "Rückgaben nach 30 Tagen sind nur mit Kaufbeleg möglich.",
  "Kontakt: support@firma.de",
  "Defekte Ware wird innerhalb der ersten 14 Tage kostenlos ersetzt."
];
const ids = docs.map((_, i) => `doc-${i}`);

// Einfügen (Embeddings erzeugt Chroma via embeddingFunction „on the fly“)
await collection.add({
  ids,
  documents: docs,
  metadatas: docs.map((_, i) => ({ src: "policy.md", i }))
});
console.log(`✅ ${docs.length} Einträge gespeichert.`);

// Erste semantische Suche
const question = "Wie lange gilt das Rückgaberecht?";
const res = await collection.query({
  queryTexts: [question],   // wichtig: queryTexts (nicht queryEmbeddings), da wir EF gesetzt haben
  nResults: 3
});
console.log("🔎 Treffer:", res.documents?.[0]);