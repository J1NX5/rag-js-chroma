import { ChromaClient } from "chromadb";

// Verbindung zu deinem lokalen Chroma-Server
const chroma = new ChromaClient({ host: "localhost", port: 8000 });

// Alle Collections holen
const collections = await chroma.listCollections();
console.log("📚 Collections:", collections.map(c => c.name));

if (collections.length === 0) {
  console.log("⚠️ Keine Collections gefunden.");
  process.exit(0);
}

// Für jede Collection: Count + Beispiel-Dokumente
for (const col of collections) {
  const coll = await chroma.getCollection({ name: col.name });
  const count = await coll.count();
  console.log(`\n🔹 Collection: "${col.name}" → ${count} Einträge`);
  
  if (count > 0) {
    const some = await coll.get({ limit: 3 });
    console.log("   📄 Beispiele:", some.documents);
  }
}