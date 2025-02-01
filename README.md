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

### Week 2: Tokens and Embeddings

Tokens and Embeddings  (Google Presentation)

Embeddings for Applications - Recommendation Systems  (https://docs.google.com/presentation/d/1wW1IOWntxAjwB1lYtQh_IVGYtUa4LiglbRUvrce3YhU/edit?usp=sharing)

Lab

Reading: 

References:  

---

### Week 3: Transformer

Transformer  (Google Presentation)

Recent Improvements in the Transformer Architecture  (Google Presentation)

Lab

Reading: 

References:  

---

### Week 4: Using Pretrained Language Models

Text Classification  (Google Presentation)

Text Clustering and Topic Modeling  (Google Presentation)

Lab

Reading: 

References:

---

### Week 5: Prompt Engineering

Intro to Prompt Engineering  (Google Presentation)

Reasoning with Generative Models  (Google Presentation)

Lab

Reading: 

References:

---

### Week 6: Semantic Search and Retrieval Augmented Generation

Semantic Search  (Google Presentation)

Retrieval Augmented Generation (RAG)  (Google Presentation)

Lab

Reading: 

References:

---

### Week 7: Midterm Quiz 1, Lab/ Research Work

---

### Week 8: Training and Fine-Tuning Language Models

Creating Text Embedding Models  (Google Presentation)

Fine Tuning Generation Models  (Google Presentation)

Lab

Reading: 

References:

---

### Week 9: Training and Fine-Tuning Language Models

Evaluating Generative Models  (Google Presentation)

Preference Tuning - RLHF, DPO  (Google Presentation)

Lab

Reading: 

References:

---

### Week 10: Multi-modal and Hybrid Retrieval

Multi-modal Retrieval  (Google Presentation)

Hybrid Search  (Google Presentation)

Lab

Reading: 

References:

---

### Week 11: Quiz 2, Lab/research work

Lab

Reading: 

References:

---

### Week 12: Graph Machine Learning
 
[1. Graph Machine Learning and Motivations](slides/1.%20Graph%20Machine%20Learning%20and%20Motivations.pdf)
 
[2. Graph Representations](slides/2.%20Graph%20Representations.pdf)

[Lab 0. Data Handling of Graphs](labs/Data%20Handling%20of%20Graphs.ipynb) 

[3. Link Analysis](slides/3.%20Link%20Analysis.pdf)

References:

[Graph Representation Learning by William L. Hamilton](https://www.cs.mcgill.ca/~wlh/grl_book/)

[TUDataset: A collection of benchmark datasets for learning with graphs](http://graphkernels.cs.tu-dortmund.de/)

[The Emerging Field of Signal Processing on Graphs](
https://arxiv.org/pdf/1211.0053.pdf)

[The Anatomy of a Large-Scale Hypertextual Web Search Engine](http://infolab.stanford.edu/~backrub/google.html)    

[Authoritative Sources in a Hyperlinked Environment](https://www.cs.cornell.edu/home/kleinber/auth.pdf)    
---

### Week 12: Graph Neural Networks

[8. Graph Neural Networks](slides/8.%20Graph%20Neural%20Networks.pdf)

[Lab 3: Hands-on GCN](labs/lab3_handson_gcn.ipynb), [Solution](workshop_solutions/lab3_handson_gcn_solution.ipynb)    
[Lab 3: Node Classification](labs/lab3_node_class.ipynb), [Solution](workshop_solutions/lab3_handson_gcn_solution.ipynb)     
[Lab 5. Graph Level Prediction](labs/lab_5_graphneuralnets_esol.ipynb), [Solution](workshop_solutions/lab_5_graphneuralnets_esol_solution.ipynb)   

[Graph Neural Networks: A review of Methods and Applications, Zhou, 2018.
Graph Representation Learning, Hamilton, 2020. ](https://arxiv.org/abs/1812.08434)

[Graph Representation Learning, Ch. 4 Graph Neural Network Model](
https://cs.mcgill.ca/~wlh/comp766/files/chapter4_draft_mar29.pdf)

---

### Week 13: Deep Generative Graph Models, Knowledge Graphs, RAG Knowledge Graphs

Workshop 5 - Deep Generative Graph Models,  illustrative notebook   
[14. Learning Deep Generative Models of Graphs.pdf](slides/14.%20Learning%20Deep%20Generative%20Models%20of%20Graphs.pdf)   
- [Deep Generative Graph Learning Notebook](labs/deep_graph_generative.ipynb)   
- [Model file](labs/model.pth)

[15. Temporal Graph Networks](slides/15.%20Temporal%20Graph%20Neural%20Networks.pdf) 

[Optional Traffice Notebook!](labs/traffic_prediction.ipynb) 


References:      
- GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models
Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, Jure Leskovec  
https://arxiv.org/abs/1802.08773
- Learning Deep Generative Models of Graphs  
https://arxiv.org/pdf/1803.03324.pdf   
- [AlphaFold](https://www.deepmind.com/research/highlighted-research/alphafold)

### Week 14: Project Work

### Week 15: Project Work

### Week 15: Final Projects Presentations

Tuesday and Thursday 15-minute boaster session for each project:
- What is your project?
- What problem does it solve?
- Why were you interested in this project?
- Brief explanation of your design and functionality.
- Status and lessons learned

[Final Project](https://docs.google.com/document/d/1X6aJZ8jhi3XlAsKcghK0CWbOqylcsl4r9vFMff2SkD0/edit?usp=sharing)    


---

