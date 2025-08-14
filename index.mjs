// index.mjs
import { ChromaClient } from "chromadb";
import { pipeline } from "@xenova/transformers";

// 1) Chroma-HTTP-Client (zeigt auf deinen lokalen Chroma-Server)
const chroma = new ChromaClient({ path: "http://localhost:8000" });

// 2) Embedding-Pipeline (läuft lokal, kein API-Key nötig)
//   Modell wird beim ersten Lauf heruntergeladen & gecached.
const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

// Helfer: Mittelwert-Pooling + L2-Normalisierung
const l2 = v => Math.hypot(...v);
const normalize = v => { const n = l2(v); return n === 0 ? v : v.map(x => x / n); };
async function embedding(text) {
  const out = await embedder(text, { pooling: "mean", normalize: false });
  return normalize(Array.from(out.data)); // -> number[]
}

async function main() {
  // 3) Collection erstellen/holen
  const collection = await chroma.getOrCreateCollection({ name: "docs" });

  // 4) Deine Daten (Beispiel)
  const docs = [
    "Unser Rückgaberecht beträgt 30 Tage.",
    "Rückgaben nach 30 Tagen sind nur mit Kaufbeleg möglich.",
    "Kontakt: support@firma.de",
    "Defekte Ware wird innerhalb der ersten 14 Tage kostenlos ersetzt."
  ];

  // 5) Embeddings berechnen und hinzufügen
  const ids = docs.map((_, i) => `doc-${i}`);
  const vectors = [];
  for (const d of docs) vectors.push(await embedding(d));

  await collection.add({
    ids,
    documents: docs,
    embeddings: vectors,
    metadatas: docs.map((_, i) => ({ src: "policy.md", idx: i }))
  });

  console.log(`✅ ${docs.length} Einträge in Chroma gespeichert.`);
  
  // (Optional) eine erste Abfrage, nur um zu prüfen, dass’s steckt:
  const q = "Wie lange gilt das Rückgaberecht?";
  const qVec = await embedding(q);
  const res = await collection.query({ queryEmbeddings: [qVec], nResults: 3 });
  console.log("🔎 Treffer:", res.documents?.[0]);
}

main().catch(err => console.error(err));