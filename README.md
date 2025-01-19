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


---

### Week 1: Intro to Information Retrieval

[Introduction to Search](https://docs.google.com/presentation/d/12Fk9c54QMsgWoO0AO8kFVG9rJQTixe352Pg0xRMJe7Y/edit?usp=sharing)  (Google Presentation)

[Information Retrieval Evaluation](https://docs.google.com/presentation/d/1IpbgmkXoMBxPws-O6DTTPWuRfzUHz_OlXre3ikuXTik/edit?usp=sharing)  (Google Presentation)

[Introduction to "Deep" Search](https://docs.google.com/presentation/d/1Wkprk_U3nMb-801gbq4Trx23VfvkYupYr43C4mrdlCM/edit?usp=sharing)  (Google Presentation)

[Lab - Search Applications](labs/Data%20Handling%20of%20Graphs.ipynb) 

Reading: 

References:  

---

### Week 2: Introduction to Generative AI, Deep Learning Fundamentals, PyTorch


[Lab - PyTorch Application](labs/Data%20Handling%20of%20Graphs.ipynb) 

---

### Week 3: Working with Datasets and Pretrained Models

[Lab - MobileNetV3](labs/Data%20Handling%20of%20Graphs.ipynb) 

---

### Week 4: Foundation Models

[Lab - Spam Classifier](labs/Data%20Handling%20of%20Graphs.ipynb) 

---

### Week 5: Adapting Foundation Models, Lightweight Fine-Tuning

[Lab - MobileNetV3](labs/Data%20Handling%20of%20Graphs.ipynb) 

---

### Week 6: Midterm Quiz 1, Lab/ Research Work

---

### Week 7: Intro to Network Analysis and Machine Learning on Graphs

[1. Graph Machine Learning and Motivations](slides/1.%20Graph%20Machine%20Learning%20and%20Motivations.pdf)
 
[2. Graph Representations](slides/2.%20Graph%20Representations.pdf)

[Lab 0. Data Handling of Graphs](labs/Data%20Handling%20of%20Graphs.ipynb) 

[Lab 1. Graph ML Research Topics](labs/Lab%201.%20Graph%20ML%20Research%20Topics.pdf)  

[Graph Laplacian Notebook](https://colab.research.google.com/github/Taaniya/graph-analytics/blob/master/Graph_Laplacian_and_Spectral_Clustering.ipynb#scrollTo=BW6RnVt1X-0Z)

References:   
[Graph Representation Learning by William L. Hamilton](https://www.cs.mcgill.ca/~wlh/grl_book/)

[TUDataset: A collection of benchmark datasets for learning with graphs](http://graphkernels.cs.tu-dortmund.de/)

[The Emerging Field of Signal Processing on Graphs](
https://arxiv.org/pdf/1211.0053.pdf)
  
### Week 8: Link Analysis and Random Walk

[3. Link Analysis](slides/3.%20Link%20Analysis.pdf)

[Lab 2: Link Analysis](labs/Lab%202.%20Link%20Analysis.pdf)  

[Lab 2: PageRank notebook](labs/PageRank.ipynb)

References:    
[The Anatomy of a Large-Scale Hypertextual Web Search Engine](http://infolab.stanford.edu/~backrub/google.html)    

[Authoritative Sources in a Hyperlinked Environment](https://www.cs.cornell.edu/home/kleinber/auth.pdf)    

### Week 9: Node Classification, Intro to Graph Neural Networks 
 
[4. Message Passing, Node Embeddings, and Representations](slides/4.%20Message%20Passing%20and%20Representations.pdf)   

[5. Node Embeddings, Random Walk, Node2vec](slides/5.%20Node%20Embeddings.pdf)
 
[Hands-on: Node2Vec](labs/DeepWalk.ipynb)  (optional)

[Lab 3: Hands-on GCN](labs/lab3_handson_gcn.ipynb)

[Lab 3: Node Classification](labs/lab3_node_class.ipynb)

### Week 10: Machine Learning Review, Graph Neural Network Intro

[6. Machine Learning Intro](slides/6.%20Machine%20Learning%20Intro.pdf)

[Hands-on Gradient Descent Notebook](labs/gradient_descent_assignment_solution.ipynb)

[Hands-on Logistic Regression with Pytorch](labs/Building%20a%20Logistic%20Regression%20Classifier%20in%20PyTorch.ipynb)

[XOR and Logistic Regression Proof](slides/XOR_and_LogisticRegression.pdf)    

[XOR and Logistic Regression Notebook](labs/XOR.ipynb)

[7. Deep Learning Intro](slides/7.%20Deep%20Learning%20Intro.pdf)

[Lab 4. Building a Neural Network in PyTorch](labs/Lab%204.%20Building%20a%20Neural%20Network%20in%20PyTorch.ipynb)

References:   

[Simplifying Graph Convolutional Networks](http://proceedings.mlr.press/v97/wu19e/wu19e.pdf)  

[SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf)  

### Week 11: Deep Graph Learning, GraphSage, Applications

[8. Graph Neural Network Intro](slides/8.%20Graph%20Neural%20Network%20Intro.pdf)  
Slides 1-16.

Workshop 5 - Graph Neural Network    
[8. Graph Neural Networks](slides/8.%20Graph%20Neural%20Networks.pdf)

[Lab 5. Graph Level Prediction](labs/lab_5_graphneuralnets_esol.ipynb)

References:

[Graph Neural Networks: A review of Methods and Applications, Zhou, 2018.
Graph Representation Learning, Hamilton, 2020. ](https://arxiv.org/abs/1812.08434)

[Graph Representation Learning, Ch. 4 Graph Neural Network Model](
https://cs.mcgill.ca/~wlh/comp766/files/chapter4_draft_mar29.pdf)
 
[Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

### Week 12: Graph Convolution Networks, Graph Relational Networks  

[9. Graph Convolution Networks](slides/9.%20Graph%20Convolution%20Networks.pdf) 

[Lab 7.1. Hands-on Loading Graphs from CSV](labs/Loading_Graphs_from_CSV.ipynb) 

[Lab 7.2. Graph ML Project](labs/Lab%207.%20Graph%20ML%20Project.pdf) 

[Optional: Additional examples of Loading Graphs from CSV](labs/tabular_to_graph.ipynb)

[10. Relational Graph Networks](slides/10.%20Relational%20Graph%20Networks.pdf)
<!--
[11. Knowledge Graph Embeddings ]() 

[12. Reasoning over Knowledge Graphs]()
-->

### Week 13: Graph Attention Networks, Recommender Systems   

[11. Graph Attention Networks](slides/11.%20Graph%20Attention%20Networks.pdf)

[Lab 7.2. Graph ML Project](labs/Lab%207.%20Graph%20ML%20Project.pdf) 

[12. Graph Recommender Systems](slides/12.%20Graph%20Recommender%20Systems.pdf)

<!--
[13. Frequent Subgraph Mining with GNNs]()

[15. Community Structure in Networks]()
-->

### Week 14: Generative Graph Models   

[13. Deep Generative Models for Graphs](slides/13.%20Generative%20Models%20for%20Graphs.pdf) 

[Lab 7.2. Graph ML Project](labs/Lab%207.%20Graph%20ML%20Project.pdf) 

[14. Learning Deep Generative Models of Graphs.pdf](slides/14.%20Learning%20Deep%20Generative%20Models%20of%20Graphs.pdf)

Hands On:   
- [Deep Generative Graph Learning Notebook](labs/deep_graph_generative.ipynb)   
- [Model file](labs/model.pth)   
- [Annimation Gif](labs/48313438-78baf000-e5f7-11e8-931e-cd00ab34fa50.gif)

References:      
- GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models
Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, Jure Leskovec  
https://arxiv.org/abs/1802.08773
- Learning Deep Generative Models of Graphs  
https://arxiv.org/pdf/1803.03324.pdf   
- [AlphaFold](https://www.deepmind.com/research/highlighted-research/alphafold)

### Week 15: Temporal Graph Models, Final Projects, Final Exam Review 

[15. Temporal Graph Networks](slides/15.%20Temporal%20Graph%20Neural%20Networks.pdf) 

[Optional Traffice Notebook!](labs/traffic_prediction.ipynb) 

### Week 15: Knowledge Graphs

### Week 15: Multimodal Search

### Week 15: Final Projects Presentations

Monday and Wednesday 5-minute boaster session for each project:
- What is your project?
- What problem does it solve?
- Why were you interested in this project?
- Brief explanation of your design and functionality.
- Status and lessons learned

[Final Project](https://docs.google.com/document/d/1X6aJZ8jhi3XlAsKcghK0CWbOqylcsl4r9vFMff2SkD0/edit?usp=sharing)    


---

