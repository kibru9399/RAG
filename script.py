from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
store = LocalFileStore('/content')

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
core_embeddings_model = HuggingFaceEmbeddings(model_name = embed_model_id)
embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings_model, store, namespace=embed_model_id
)
#vector_store = FAISS.from_documents(data_docs, embedder)
import torch
import transformers, accelerate

model_id = "meta-llama/Llama-2-13b-chat-hf"
token = "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"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(
    model_id
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    #token = token, 
    #use_auth_token=token
)

model.eval()
tokenizer = tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id
)

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=True,
    temperature=0.01,
    max_new_tokens=256
)
from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)
retriever = vector_store.as_retriever()

from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
handler = StdOutCallbackHandler()
qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)
qa_with_sources_chain({"query" : "based on the reviews, what is cheap but also durable furniture?"})