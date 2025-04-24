CSC 5201 - Search machine learning: Course on applying LLM and Graph Machine Learning to search. (4 Credits)
----

### Course Description

The purpose of this course is to explore current state-of-the-art AI methods to enhance search applications.

An emphasis will be placed on generative learning models and large language modeling, and will 
include retrieval augmented generation, multi-modal retrieval, and graph machine learning to capture rich relations in data.

Class is taught as a graduate “topics” style course.  Instructor provides foundational knowledge and course structure. 
Students help research course topics and contribute their finding and knowledge to the class.

Our work will cast a “wide net” across standard and current state-of-the-art information retrieval 
methods to identify the most promising areas for deeper exploration based on student and instructor interest.

### Course Topics

- Introduction to search
- Foundation models
- Large language models
- Generative machine learning
- Keyword search and sparse information retrieval.
- Graph based retrieval.
- Embeddings and dense retrieval.
- Hybrid search integrating sparse, graph, and dense retrieval.
- Large Language Model reranking
- Retrieval Augmented Generation
- Graph Machine Learning for learning relational embeddings
- Learning knowledge graphs
- Enhancing search with knowledge graphs
- Applications

### Course Learning Outcomes

Upon successful completion of this course, the student will be able 
to: TBD

### Prerequisites by Topic

- Version control / Git
- Python programming
- Introductory machine learning course

Grading (no final exam)  
Weekly labs and research assignments: 40%  
Final project: 30%   
Midterm Quiz 1: 15%   
Midterm Quiz 2: 15%   

Class will be structured in 3 parts: 
- Search, Generative AI, LLM
- Graph Machine Learning, RAG  
- Advanced topics, multi-modal retrieval and final project.

Office DH425  
TBD: As needed

References:  

[Graph Representation Learning by William L. Hamilton](https://www.cs.mcgill.ca/~wlh/grl_book/)

[http://www.cs.cornell.edu/home/kleinber/networks-book/Networks, Crowds, and Markets: Reasoning About a Highly Connected World by David Easley and Jon Kleinberg](http://www.cs.cornell.edu/home/kleinber/networks-book/)

[Network Science by Albert-László Barabási](http://networksciencebook.com/)

[Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges
Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković](https://arxiv.org/pdf/2104.13478)

[Geometric Deep Learning](https://geometricdeeplearning.com/lectures/)

[Stanford Machine Learning on Graphs](http://web.stanford.edu/class/cs224w/)

[PyG - Pytorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest)  

[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)

https://www.pinecone.io/learn/

https://cohere.com/developers 

https://platform.openai.com/docs/tutorials 

[Hands On Large Language Models](https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961/ref=sr_1_1?dib=eyJ2IjoiMSJ9.XmrBJHOrpr0Bp3aEocU10Q6AoT-tytNAqxrij7t7-2KnBn7or0WxbkSM2R6X31C0t-LkWxRUaZMqaixz_NCdWv2Zr18h7U9EDhgVtdUZa1_TAG3dEaXXstt1BhCDx4YovwDPCArm_8Kwx8yp2Dlci_Hhp0lp5AAu9ul38KBwtv15hGBMgN4ILmujZuYKAdbj-O7MUeba4lVq5q7vUiPu35Qx5YFIpGxnN_Sr3wwkic0.AghCsQzPJgAJ2SYSe371MrKd4WJC-SwriFY2vwYrR_o&dib_tag=se&hvadid=680640551436&hvdev=c&hvlocphy=9018766&hvnetw=g&hvqmt=e&hvrand=15152233938042636768&hvtargid=kwd-2098329602793&hydadcr=19108_13375724&keywords=hands+on+large+language+models&qid=1737401091&sr=8-1), Alamar and Grootendorst


---

### Week 1: Intro to Information Retrieval

[Introduction to Search](https://docs.google.com/presentation/d/12Fk9c54QMsgWoO0AO8kFVG9rJQTixe352Pg0xRMJe7Y/edit?usp=sharing)  (Google Presentation)

[Information Retrieval Evaluation](https://docs.google.com/presentation/d/1IpbgmkXoMBxPws-O6DTTPWuRfzUHz_OlXre3ikuXTik/edit?usp=sharing)  (Google Presentation)

[Introduction to "Deep" Search](https://docs.google.com/presentation/d/1Wkprk_U3nMb-801gbq4Trx23VfvkYupYr43C4mrdlCM/edit?usp=sharing)  (Google Presentation)

[Lab 1 - Awesome Search](https://docs.google.com/document/d/1U5bncs0CPutVTKW0ZhjM6loCTbythBI0M2KaSoVBGfk/edit?usp=sharing)

Reading: 

References:  
Introduction to Information Retrieval. Ch. 1, 2, 6 & 8.      
[Evaluation Measures in Information Retrieval](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf)

---

### Week 2: Introduction to Language Models

[Introduction to Language Models](https://docs.google.com/presentation/d/1WRJr3aPW-KrB1Wu0CP6H2a8AyKcHtlPuDCch-CgnP7M/edit?usp=sharing)

[Transformer Model Intro](https://docs.google.com/presentation/d/1wW1IOWntxAjwB1lYtQh_IVGYtUa4LiglbRUvrce3YhU/edit?usp=sharing)

[Lab 2 - Language Models](labs/language_model.ipynb)

Reading: 

[Attention Is All You Need Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. 2018](https://arxiv.org/abs/1706.03762)

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova](https://arxiv.org/abs/1810.04805)

[Improving Language Understanding by Generative Pre-Training](https://gwern.net/doc/www/s3-us-west-2.amazonaws.com/d73fdc5ffa8627bce44dcda2fc012da638ffb158.pdf)

OpenAI, “Gpt-4 technical report.” arXiv preprint arXiv:2303.08774 (2023). https://arxiv.org/abs/2303.08774

[DeepSeek FAQ](https://stratechery.com/2025/deepseek-faq/)   


References:  
[Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g), Andrej Karpathy

---

### Week 3: Transformer, Using Pretrained LLMs

[Transformer Model](https://docs.google.com/presentation/d/1wW1IOWntxAjwB1lYtQh_IVGYtUa4LiglbRUvrce3YhU/edit?usp=sharing)

[Lab 3 - Embeddings](labs/embeddings.ipynb)

[Pretrained LLMs for Text Classification](https://docs.google.com/presentation/d/18t2Tk7ywAp1Zpk70sUpI-KqB1QiBs4hSHZibS-OkZL4/edit?usp=sharing)

Required Viewing:
[Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g), Andrej Karpathy.

Reading:

[Attention Is All You Need Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. 2018](https://arxiv.org/abs/1706.03762)

[Improving Language Understanding by Generative Pre-Training](https://gwern.net/doc/www/s3-us-west-2.amazonaws.com/d73fdc5ffa8627bce44dcda2fc012da638ffb158.pdf)


References:  

---

### Week 4: Retrieval Augmented Generation, Fine Tuning

[RAG](https://docs.google.com/presentation/d/1jeRlp1Rs4DiqJzgD0iAoSmNi8PBMQgC-_g55raUWLi8/edit?usp=sharing)

[Lab 4 - RAG](labs/lab_4_RAG.ipynb)

[Fine Tuning](https://docs.google.com/presentation/d/1kXCwsgEKVP1HpNnWwH0qvShAzPaeBrEf0HOwBwf-wTw/edit?usp=sharing)

Reading:   
[Databrick Guide to LLMs](references/compact-guide-to-large-language-models.pdf)

---

### Week 5: Fine Tuning, Hybrid Search

[Fine Tuning](https://docs.google.com/presentation/d/1kXCwsgEKVP1HpNnWwH0qvShAzPaeBrEf0HOwBwf-wTw/edit?usp=sharing)

[Lab 5: Fine-tuning-QLoRA](https://colab.research.google.com/drive/1MCEZALkIBfpAwpB2qeXLMjbKgp4fWziR?usp=sharing)

[Lab 6: RAG Search (2-weeks)](https://docs.google.com/document/d/1L3HWRjvcHH5ktssk66YgmTgHNj7QVBoZOqVaQbCBL9s/edit?usp=sharing)

[Hybrid Search](https://docs.google.com/presentation/d/1irn32NDVZRn4iOEQcRYwFPYI5V2_QE2THxpNXaIvUp4/edit?usp=sharing)

[Hybrid Search Demo Notebook](labs/hyrbrid_search_pinecone.ipynb)

---

### Week 6: Quiz, Graph Machine Learning Intro

[Quiz 1](https://docs.google.com/document/d/1uHQHKn8ITFDoj8bi89mCap_HN_vJkt0UZDDJ704DV6Y/edit?usp=sharing)

Lab: RAG Lab Continued

[Graph Machine Learning](https://docs.google.com/presentation/d/10PEVzMoWOZgEbQNruZDYYHaviVK_33z5ZEAKlRr5O84/edit?usp=sharing)

[Graph Representations](https://docs.google.com/presentation/d/1VQCcRT64oOcflhjqFmsefhQQIx3mi59Sa7rFMVYFxOg/edit?usp=sharing)

[Data Handling of Graphs](labs/Data Handling of Graphs.ipynb)

References:  
See graph references above.

---

### Week 7: Link Analysis

RAG Lab Review

[Link Analysis](https://docs.google.com/presentation/d/14Kfi4hHJnjHTrreEo8UoiEsssFQi6w6WskxjwJNvzq8/edit?usp=sharing)

[Lab Link Analysis](labs/Lab%202.%20Link%20Analysis.pdf)

[Lab Page Rank Notebook](labs/PageRank.ipynb) 

[Message Passing and Representations](https://docs.google.com/presentation/d/1ZEbDkmSfwZq3qs7damgC4Z8oUChKSNq9h76wCSxB7fI/edit?usp=sharing)

Reading:

The Anatomy of a Large-Scale Hypertextual Web Search Engine

Authoritative Sources in a Hyperlinked Environment

---

### Week 8:  Node2Vec, Graph Neural Networks

[Node Embeddings, DeepWalk, Node2Vec](https://docs.google.com/presentation/d/110t10GtYP26WkU4xV3lwsZ0N1gyl45uInlJ9ADotmzE/edit?usp=sharing)

Notebook:  
[Hands-on: Node2Vec](labs/Node2Vec.ipynb) 

[Graph Neural Networks](https://docs.google.com/presentation/d/1EC7-okKsZnDIygkVBlar7udeehdmklrOIim7oObVaYI/edit?usp=sharing)

Notebooks:   
[Hands-on GCN](labs/lab3_handson_gcn.ipynb)

---

### Week 9: Graph Convolution Networks, GraphSage

[Graph CNN, GraphSage](https://docs.google.com/presentation/d/1NDVedsdYbMsFQcBprrh8dk3_3MM6NgSEW-eQ56xzwkI/edit?usp=sharing)

Notebooks:   
[Node Classification](labs/lab3_node_class.ipynb)  

[Graph Level Prediction](labs/lab_5_graphneuralnets_esol.ipynb)

[Graph Attention Networks](https://docs.google.com/presentation/d/1C5qbd6ev3Z4iRybvyK8OIPQTxgQzmVfcXZNGEkuFXT8/edit?usp=sharing

[Graph Recommender Systems](https://docs.google.com/presentation/d/1950l6M3aYauDPLWcmdxPd6gHCQasT6mi9tWZJ6YDrsM/edit?usp=sharing)

Notebooks:  

[Amazon Recommender](labs/AMZN_Recommender_Urbain.ipynb)

---

### Week 10: Deep Generative Graph Models

Deep Generative Graph Models,  illustrative notebook   

[Generative Models for Graphs](https://docs.google.com/presentation/d/1z7Lwtad0NtIK3EGEVdyLqhsVh4JTOBhlRhTSz_zNw4A/edit?usp=sharing)     

Notebook:   
- [Deep Generative Graph Learning Notebook](./labs/deep_graph_generative.ipynb)   
- [Model file](./labs/model.pth)
- [Animation GIF](./labs/48313438-78baf000-e5f7-11e8-931e-cd00ab34fa50.gif)

[Temporal Graph Networks](https://docs.google.com/presentation/d/1Wvi37x7OtedLAsNYUkb7XsEuU6dDnQOyWovQjPytbLk/edit?usp=sharing) 

[Traffic Notebook - Temporal GCN](labs/traffic_prediction.ipynb) 


References:      
- GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models
Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, Jure Leskovec  
https://arxiv.org/abs/1802.08773
- Learning Deep Generative Models of Graphs  
https://arxiv.org/pdf/1803.03324.pdf   
- [AlphaFold](https://www.deepmind.com/research/highlighted-research/alphafold)

### Week 11: Graph Machine Learning Quiz, Final Projects

[Graph Machine Learning Quiz](https://docs.google.com/document/d/1-q7OfqqcLomK1be7yrurZ4Lcn7OxuW89_SisT95AeBo/edit?usp=sharing)

[Final Project Proposal](https://docs.google.com/document/d/1lrhR7Ahy4HY4h6I5NTsvwiK-2-0OHi51_VXi3Ahs4rw/edit?usp=sharing)


### Week 12: Knowledge Graphs, RAG Knowledge Graphs

[Final Project Proposal Presentation](https://docs.google.com/document/d/1lrhR7Ahy4HY4h6I5NTsvwiK-2-0OHi51_VXi3Ahs4rw/edit?usp=sharing)    
Tuesday and Thursday 15-20 minute boaster session for each project:
- What is your project?
- What problem does it solve?
- Why were you interested in this project?
- Brief explanation of tools and resources needed for your project  
- Brief explanation of your design and functionality.
- Exemplar projects

Slides: [Knowledge Graphs, Graph ML, and LLMs](https://docs.google.com/presentation/d/1CmzOCP6nFoDuHUpELfPXmqFqzAGNWugrinsOD-ueVl0/edit?usp=sharing)

Instructions: [Lab: Constructing Knowledge Graph with LLM Extraordinaire](https://docs.google.com/document/d/13xrIFY1JaBc-mzMBmLSFYqTX6EsSM6Msa-EpELWZN78/edit?usp=sharing)

Notebook: [Constructing a Knowledge Graph](./labs/constructing_knowledge_graph.ipynb)

Notebook: [Creating a Knowledge Graph Embedding](labs/pykeen_knowledge_graph_embedding.ipynb)

### Week 13: Multimodal RAG, Agentic RAG

[Multimodal RAG](https://docs.google.com/presentation/d/15PG0IwOVFuEz4EjissFCSQc8YXT9gHE9-7h7MoHg6iI/edit?usp=sharing)

Notebook: [Multimodal Retrieval Augmented Generation](labs/Multimodal_RAG_VoyageAI.ipynb)


[Agentic RAG](https://docs.google.com/presentation/d/1tcqH2DkXXc-K8sVpzuTRgSw7ScDcnBRVzvy2i275Jp4/edit?usp=sharing)

Notebooks:  
Workflow (optional) -    
[Basic Workflows](labs/basic_workflows.ipynb)    
[Orchestrator Workers](labs/orchestrator_workers.ipynb)     
[Evaluator Optimizer](labs/evaluator_optimizer.ipynb)     

Agentic (required) -     
[Build LLM Agent combining Reasoning and Action](labs/ReAct_Example.ipynb)

### Week 14: Advanced Topic or Project Work

LLM Knowledge Graph RAG Application

Project work

### Week 15: Final Project Presentations

Tuesday and Thursday 15-20 minute boaster session for each project:
- What is your project?
- What problem does it solve?
- Why were you interested in this project?
- Brief explanation of tools and resources needed for your project   
- Exemplar projects
- Brief explanation of your design and functionality.
- Status and lessons learned

[Final Project Proposal](https://docs.google.com/document/d/1lrhR7Ahy4HY4h6I5NTsvwiK-2-0OHi51_VXi3Ahs4rw/edit?usp=sharing)


---

