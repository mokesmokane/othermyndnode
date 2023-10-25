import express from 'express';
import admin from 'firebase-admin';
import { FieldPath } from '@google-cloud/firestore';
import dotenv from 'dotenv';
import cors from 'cors';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { EPubLoader } from "langchain/document_loaders/fs/epub";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
const openaiApiKey = process.env.OPENAI_API_KEY;
const pineconeApiKey = process.env.PINECONE_API_KEY;
const pineconeEnvironment = process.env.PINECONE_ENVIRONMENT;
const openaiOrg = process.env.OPENAI_API_ORG;

const openai = new OpenAI({
  organization: openaiOrg,
  apiKey: openaiApiKey,
});


dotenv.config();

const app = express();
const port = 3000;

app.use(cors());

const upload = multer({ dest: 'uploads/' });
const firebaseConfig = {
  type: process.env.FIREBASE_TYPE,
  projectId: process.env.FIREBASE_PROJECT_ID,
  privateKeyId: process.env.FIREBASE_PRIVATE_KEY_ID,
  privateKey: process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, '\n'),
  clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
  clientId: process.env.FIREBASE_CLIENT_ID,
  authUri: process.env.FIREBASE_AUTH_URI,
  tokenUri: process.env.FIREBASE_TOKEN_URI,
  authProviderX509CertUrl: process.env.FIREBASE_AUTH_PROVIDER_X509_CERT_URL,
  clientX509CertUrl: process.env.FIREBASE_CLIENT_X509_CERT_URL
};

admin.initializeApp({
  credential: admin.credential.cert(firebaseConfig)
});


const db = admin.firestore();
const pinecone = new Pinecone({
  apiKey: pineconeApiKey,
  environment: pineconeEnvironment,
});

async function getPdfBits(filePath) {
  const loader = new PDFLoader(filePath, {splitPages: true});

  const docs = await loader.load();
  // PDF processing and Firebase Firestore logic here
  const splitter = new RecursiveCharacterTextSplitter({       
      chunkSize: 1000,
      chunkOverlap: 200,
      separators: ["\n\n","\n","\r\n","\r","\t"," "],
    });
  const segments = await splitter.splitDocuments(docs);
  
  let updated = [];
  let pages = {};

  segments.forEach(segment => {
    const page = segment.metadata.loc.pageNumber;
    if (!pages[page]) {
      pages[page] = [];
    }
    
    delete segment.metadata.pdf.metadata;

    pages[page].push(segment);
  });

  let firestorePages = [];

  Object.keys(pages).forEach(page => {
    firestorePages.push({ page: page, segments: [] });
    pages[page].forEach((segment, index) => {
      segment.metadata["index"] = index;
      var update = {
        text: segment.pageContent,
        metadata: segment.metadata
      };
      firestorePages[firestorePages.length - 1].segments.push(update);
      updated.push(update);
    });
  });
  return [firestorePages,updated];
}

async function getEpubBits(filePath, fileName) {
  const loader = new EPubLoader(filePath, {splitChapters: true});

  const docs = await loader.load();
  
  const splitter = new RecursiveCharacterTextSplitter({       
      chunkSize: 1000,
      chunkOverlap: 200,
      separators: ["\n\n","\n","\r\n","\r","\t"," "],
    });
  console.log(docs.length)
  for (let i = 0; i < docs.length; i++) {
    console.log(docs[i].metadata)
  }
  const segments = await splitter.splitDocuments(docs);
  console.log(segments[0]);
  
  let updated = [];
  let pages = {};
  let firestorePages = [];
  const chunkSize = 5;
  for (let i = 0; i < segments.length; i += chunkSize) {

    const chunk = segments.slice(i, i + chunkSize);
    const pageNumber = i/chunkSize;
    const pagesegments = chunk.map((segment, index)=>{
      
      let meta =  {
        loc: segment.metadata.loc,
        filename: fileName,
        chunk: index
      }
      var update = {page: pageNumber, text: segment.pageContent, metadata: meta};
      updated.push(update);
      return update;
    });
    firestorePages.push({ page: pageNumber, segments: pagesegments });
  }

  return [firestorePages,updated];
}

async function uploadKnowledge(req, res) {
  
  const user = req.params.user
  if (!req.file) {
    return res.status(400).send("No file uploaded.");
  }

  const filePath = req.file.path;
  const fileName = req.file.originalname;
  let firestorePages,updated = ([],{});
  if (fileName.endsWith(".pdf")) {
    [firestorePages,updated] = await getPdfBits(filePath);
  } else if(fileName.endsWith(".epub")){
    [firestorePages,updated] = await getEpubBits(filePath, fileName);
  }
  
  
  let guid = uuidv4();

  let idmap_ref = db.collection('UsersProd').doc(user).collection('PDFs').doc("idMap");
  let idmap = await idmap_ref.get();
  if (idmap.exists) {
    console.log("idmap does exist")
    let mymap = idmap.data().map
    if(mymap[fileName]){
      console.log("already exist!!!!!!!!!!!!!!!!!!!!!!!!")
      res.json({ status: "success", message: "already exists" });
      return;
    }
  }
  

  await Promise.all(firestorePages.map(fPage => {
    const pageRef = db.collection('UsersProd').doc(user)
                      .collection('PDFs').doc(guid)
                      .collection('pages').doc(`${fPage.page}`);
    return pageRef.set(fPage);
  }));

  async function processEmbeddings() {
    const chunkSize = 50;
    for (let i = 0; i < updated.length; i += chunkSize) {
      const chunk = updated.slice(i, i + chunkSize);
      const metas = chunk.map((segment, index) => ({
        page: segment.metadata["page"],
        filename: fileName,
        chunk: index
      }));
  
      const texts = chunk.map(segment => segment.text);
      const ids = texts.map((_, index) => `${guid}_chunk_${i / chunkSize}_${index}`);
    
      const embeddingResponse = await openai.embeddings.create({
        input: texts,
        model: "text-embedding-ada-002"
      });

      let embeds = embeddingResponse.data.map(record => record.embedding);
      const toUpsert = ids.map((id, index) => ({ id:id, values: embeds[index], metadata: metas[index] }));
      let pindex = pinecone.index('external-index').namespace(guid);
      pindex.upsert(toUpsert); 
    
    }
  }
  
  await processEmbeddings();
  if (idmap.exists) {
    let mymap = idmap.data().map
    mymap[fileName] = guid
    await idmap_ref.update({map:mymap})
  }else{
    console.log("idmap does not exist!!!!!!!!!!!!!!!!!!!!!!!!")
    await idmap_ref.set({map: {[fileName]: guid}})
  }
  console.log("PDF processed");
  res.json({ status: "success", message: "PDF processed" });
}

app.delete("/delete/:namespace" , async (req, res) => {
  const guid = req.params.namespace
  let pindex = pinecone.index('external-index').namespace(guid);
  pindex.deleteAll();
  res.json({"status": "success"});
});

app.post("/upload/pdf/:user", upload.single('pdf'), uploadKnowledge);

app.post("/upload/epub/:user", upload.single('epub'), uploadKnowledge);
//ping endpoint:
app.get("/ping", (req, res) => {
  res.json({ status: "success", message: "pong" });
});

const PORT = process.env.PORT || 3000
app.listen(PORT, () => {
  console.log(`Server is running on https://othermyndnode.herokuapp.com:${PORT}`);
});
