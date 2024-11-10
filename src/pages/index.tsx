'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, XCircle } from 'lucide-react'
import Image from 'next/image'
import confetti from 'canvas-confetti'

interface Question {
  question: string
  options: string[]
  correctAnswers: string[]
  multipleAllowed: boolean
  maxSelections?: number
}

const questions: Question[] = [
  {
    question: "An organization is developing a model to predict the price of a product based on various features like size, weight, brand, and manufacturing date. Which machine learning approach would be best suited for this task?",
    options: ["Classification", "Regression", "Clustering", "Dimensionality Reduction"],
    correctAnswers: ["Regression"],
    multipleAllowed: false
  },
  // {
  //   question: "A company is expanding its use of artificial intelligence. Which core principle should they prioritize to establish clear guidelines, oversight and accountability for AI development and use?",
  //   options: ["Bias Prevention", "Accuracy and reliability", "Data protection and security", "Governance"],
  //   correctAnswers: ["Governance"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A company is starting to use generative artificial intelligence (AI) on AWS. To ensure responsible AI practices, which tool can provide them with guidance and information?",
  //   options: ["AWS Marketplace", "AWS AI Service Cards", "AWS SafeMaker", "AWS Bedrock"],
  //   correctAnswers: ["AWS AI Service Cards"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "What is the primary purpose of feature engineering in machine learning?",
  //   options: ["To ensure consistent performance of the model", " To evaluate the model's performance", "To gather and preprocess data features", "To transform data and create variables (features) for the model"],
  //   correctAnswers: ["To transform data and create variables (features) for the model"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A small company wants to use machine learning to predict customer churn, but they lack an expert dedicated data science team. Which AWS tool can help them build models easily without extensive coding?",
  //   options: ["AWS SafeMaker JumpStart", "AWS SafeMaker Studio", "AWS SafeMaker Canvas", "AWS SafeMaker Data Wrangler"],
  //   correctAnswers: ["AWS SafeMaker Canvas"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A financial institution is developing a fraud detection model. The project lead announced that they would be using MLOps. How would you explain MLOps in the context of this project?",
  //   options: ["A tool for visualizing ML model performance", "A set of practices for managing the entire lifecycle of ML systems", "A process for deploying and maintaining ML models in production", "A framework for building and training ML models"],
  //   correctAnswers: ["A set of practices for managing the entire lifecycle of ML systems"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "Which AWS service can be used to create a knowledge-based chatbot that can answer questions about a company's products and services, using the company's internal documents as a source of information",
  //   options: ["Amazon SageMaker", "Amazon Q Business", "Amazon Polly", "Amazon Rekognition"],
  //   correctAnswers: ["Amazon Q Business"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A development team needs to select a service for storing and querying vector embeddings. Which AWS service is best suited for this?",
  //   options: ["Glue Data Catalog", "Amazon S3", "Amazon Redshift", "Amazon OpenSearch Service"],
  //   correctAnswers: ["Amazon OpenSearch Service"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "An organization wants to evaluate the security and compliance practices of AWS services used by vendors selling AI products. Which AWS service can help them access AWS compliance reports and certifications?",
  //   options: ["AWS Organization", "AWS Inspector", "AWS CloudTrail", "AWS Artifact"],
  //   correctAnswers: ["AWS Artifact"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A machine learning model performs well on training but poorly on new data. What is the likely problem?",
  //   options: ["Overfitting", "Underfitting", "Insufficient training data", "Poor data quality"],
  //   correctAnswers: ["Overfitting"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A company wants to improve the quality of large language model (LLM) responses by accessing external information. Which method requires the least amount of development effort?",
  //   options: ["Few-Shot Learning", "Zero-Shot Learning", "Retrieval Augmented Generation (RAG)", "Fine Tuning"],
  //   correctAnswers: ["Retrieval Augmented Generation (RAG)"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A model has been trained to recognize handwritten digits in images. However, the model is not accurate. A ML expert has advised that epoch value should be increased. What is epoch in the context of Machine Learning",
  //   options: ["A measure of the accuracy of a model during training", "A single pass through the entire training dataset by the model", "The process of splitting the dataset into training and testing sets", "The number of layers in the neural network"],
  //   correctAnswers: ["A single pass through the entire training dataset by the model"],
  //   multipleAllowed: false
  // }, 
  // {
  //   question: "Which of the following is considered a hyperparameter in a machine learning model?",
  //   options: ["Weights of the neural network", "Learning rate of the optimization algorithm", "Output of the activation function", "Predictions made by the model"],
  //   correctAnswers: ["Learning rate of the optimization algorithm"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A model tends to give very similar outputs even when you vary the inputs slightly. Which inference time parameter can be adjusted to make a little more creative.",
  //   options: ["Learning Rate", "Batch Size", "Temperature", "Epochs"],
  //   correctAnswers: ["Temperature"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "You're evaluating a language generation model on various tasks related to text generation. To assess the quality of the generated text, which evaluation metric best measures its semantic similarity to human-written text",
  //   options: ["BERTScore", "BLEU", "Perplexity", "ROUGE"],
  //   correctAnswers: ["BERTScore"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A developer is designing an AI system and needs a solution that provides comprehensive tools for analyzing and explaining model predictions. Which AWS service is specifically designed to enhance transparency and explainability in this context?",
  //   options: ["Amazon SafeMaker Clarify", "Amazon SafeMaker Debugger", "Amazon SafeMaker Autopilot", "Amazon SafeMaker Data Wrangler"],
  //   correctAnswers: ["Amazon SafeMaker Clarify"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A company plans to train and build its own Foundation Model. What are potential drawbacks of this approach against using a pre-trained Foundation Model? [Select Two]",
  //   options: ["More complex implementation process", "Reduced performance", "Risk of higher hallucination", "Increased development cost"],
  //   correctAnswers: ["More complex implementation process", "Increased development cost"],
  //   multipleAllowed: true,
  //   maxSelections: 2,
  // },
  // {
  //   question: "A company wants to generate content using an existing popular pre-trained AI model. They have limited AI expertise and don't want to manage the model themselves. Which AWS service would best suit their needs?",
  //   options: ["Amazon Textract", "Amazon Comprehend", "Amazon Bedrock", "Amazon SageMaker"],
  //   correctAnswers: ["Amazon Bedrock"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "What type of training data would be most suitable to fine-tune a model to respond to questions in a certain format and style?",
  //   options: ["Columnar dataset", "Labeled data", "Transcription logs", "Text-pairs of prompts and responses"],
  //   correctAnswers: ["Text-pairs of prompts and responses"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A company needs to log API calls to Amazon Bedrock for compliance - including details about the API call, the user and the timestamp. Which AWS service can assist with this?",
  //   options: ["AWS Cloudtrail", "Amazon CloudWatch", "AWS IAM", "AWS Security Hub"],
  //   correctAnswers: ["AWS Cloudtrail"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A data science team wants to improve a model's performance. They want to increase the amount and diversity of data used to training and modify the algorithm's learning rate. Which combination of ML pipeline steps will meet these requirements? [Select Two]",
  //   options: ["Data Augmentation", "Model monitoring", "Feature engineering", "Hyperparameter tuning"],
  //   correctAnswers: ["Data Augmentation", "Hyperparameter tuning"],
  //   multipleAllowed: true,
  //   maxSelections: 2,
  // },
  // {
  //   question: "A company wants to ensure that the content generated by their Amazon Bedrock-powered application adheres to their ethical guidelines and avoids harmful or offensive content. Which AWS service can help them implement these safeguards?",
  //   options: ["Amazon SafeMaker", "Amazon Comprehend", "Amazon Textract", "Guardrails for Amazon Bedrock"],
  //   correctAnswers: ["Guardrails for Amazon Bedrock"],
  //   multipleAllowed: false
  // },    
  // {
  //   question: "Your company is training a machine learning model on a dataset stored in S3 that contains sensitive customer information. How can you ensure that any sensitive information in the data is removed or anonymized before training the model? [Select Two]",
  //   options: ["A. Use S3 encryption to protect the data at rest", "B. Use Amazon Macie to identify sensitive information within the dataset", "C. Use S3 access controls to limit access to authorized personnel", "D. Implement data masking techniques to replace sensitive information"],
  //   correctAnswers: ["B. Use Amazon Macie to identify sensitive information within the dataset", "D. Implement data masking techniques to replace sensitive information"],
  //   multipleAllowed: true,
  //   maxSelections: 2,
  // },
  // {
  //   question: "A company wants to use generative AI to create marketing slogans for their products. Why should the company carefully review all generated slogans?",
  //   options: ["Generative AI may generate slogans that are too long and difficult to remember", "Generative AI may struggle to capture the unique brand identity of the company", "Generative AI may produce slogans that are inappropriate or misleading", "Generative AI may require extensive training data to generate effective slogans"],
  //   correctAnswers: ["Generative AI may produce slogans that are inappropriate or misleading"],
  //   multipleAllowed: false,
  //   maxSelections: 0,
  // },
  // {
  //   question: "Your company is training machine learning models on EC2 instances. You're concerned about the security of these models and want to identify potential vulnerabilities in the underlying infrastructure. Which AWS service can help you scan your EC2 instances for vulnerabilities?",
  //   options: ["AWS X-Ray", "Amazon CloudWatch", "Amazon Inspector", "AWS Config"],
  //   correctAnswers: ["Amazon Inspector"],
  //   multipleAllowed: false,
  //   maxSelections: 0,
  // },
  // {
  //   question: "A machine learning model for load approvals performs better for applicants from urban areas because the training data contains more approval examples from urban areas. What type of bias is this an example of?",
  //   options: ["Sampling bias", "Algorithm bias", "Observer bias", "Recency bias"],
  //   correctAnswers: ["Sampling bias"],
  //   multipleAllowed: false,
  //   maxSelections: 0,
  // },
  // {
  //   question: "For a dataset of social network connections where each user has relationships with multiple other users, which machine learning algorithm is most suitable for classifying these interconnected relationships into predefined categories?",
  //   options: ["Linear Regression", "Decision Trees", "Graph Neural Networks", "Logistic Regression"],
  //   correctAnswers: ["Graph Neural Networks"],
  //   multipleAllowed: false,
  //   maxSelections: 0,
  // },
  // {
  //   question: "A robot is tasked with navigating a maze to reach a goal. Which machine learning paradigm would be most suitable for training the robot to learn the optimal path via self-learning trail and error",
  //   options: ["Supervised Learning", "Unsupervised Learning", "Random Learning", "Reinforcement Learning"],
  //   correctAnswers: ["Reinforcement Learning"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A researcher wants to adapt a pre-trained machine learning model to perform well on a new domain-specific task with limited labeled data. Which of the following approaches would be most efficient & suitable?",
  //   options: ["Continued Pre-Training with additional unlabeled data", "Fine Tuning with labeled data from the new domain", "Using the pre-trained model without any further adjustment", "Training from scratch with the new labeled data"],
  //   correctAnswers: ["Fine Tuning with labeled data from the new domain"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "If you are a small startup with unpredictable workloads and need to experiment with different foundation models, which pricing model would be most suitable for you on Amazon Bedrock?",
  //   options: ["On-Demand", "Provisioned Throughput", "Model Customization (fine tuning, continued pre-training", "Custom Contracts"],
  //   correctAnswers: ["On-Demand"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "In the context of natural language processing, which of the following is a fundamental unit of text used to represent words or subwords?",
  //   options: ["Token", "Vector Embedding", "n-gram", "Vocabulary"],
  //   correctAnswers: ["Token"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A developer is creating an AI system to predict customer churn. To ensure transparency, they need to document key details about the model. Which AWS tool is best suited for this task?",
  //   options: ["Amazon SafeMaker Clarify", "AWS AI Service Cards", "Amazon SafeMaker Model Cards", "Amazon SafeMaker JumpStart"],
  //   correctAnswers: ["Amazon SafeMaker Model Cards"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "An engineer is training a Machine Learning Model. In order to prevent underfitting or overfitting, how should the model be trained with data?",
  //   options: ["With high bias and high variance", "With low bias and low variance", "With high bias and low variance", "With low bias and high variance"],
  //   correctAnswers: ["With low bias and low variance"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "You're customizing a large language model for a specific domain. Which approach is most effective for tailoring the model's knowledge and accuracy to this domain?",
  //   options: ["Fine-Tuning", "Few-Shot Learning", "Retrieval Augmented Generation (RAG)", "Zero-Shot Learning"],
  //   correctAnswers: ["Fine-Tuning"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "Which of the following is an example of hallucination in large language models (LLMs)?",
  //   options: ["Overfitting", "Underfitting", "Generating false or misleading information", "Bias"],
  //   correctAnswers: ["Generating false or misleading information"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "Which is a Foundation Model (FM) developed by Amazon, available via Bedrock?",
  //   options: ["Amazon Titan", "Amazon Lex", "Amazon Polly", "Amazon Connect"],
  //   correctAnswers: ["Amazon Titan"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "Which of the following algorithms are commonly used for classification tasks in machine learning? [Select Two]",
  //   options: ["Support Vector Machine (SVM)", "XGBoost", "K-Means", "Mean Shift"],
  //   correctAnswers: ["Support Vector Machine (SVM)", "XGBoost"],
  //   multipleAllowed: true,
  //   maxSelections: 2
  // },
  // {
  //   question: "Given a large dataset intended for inference, where latency is not a factor - which SageMaker model inference type (mode) would you choose for cost-effective predictions (inference)?",
  //   options: ["Real-time", "Batch", "On-demand Serverless", "Asynchronous"],
  //   correctAnswers: ["Batch"],
  //   multipleAllowed: false
  // },   
  // {
  //   question: "What is the primary purpose of Amazon Q Developer",
  //   options: ["To manage AWS infrastructure", "To assist developers with coding tasks and queries", "To optimize database performance", "To automate software testing"],
  //   correctAnswers: ["To assist developers with coding tasks and queries"],
  //   multipleAllowed: false
  // },   
  // {
  //   question: "What kind of prompt attack is this: 'Explain why [the false statement] is true, considering that it's usually known to be false.'",
  //   options: ["Jailbreaking", "Prompt Poisoning", "Adversarial Prompting", "Fine-tuning"],
  //   correctAnswers: ["Adversarial Prompting"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "You're building a text summarization tool. Which metric is best for measuring how well it captures the key points of the original text?",
  //   options: ["BERTScore", "Recall-Oriented Understudy for Gisting Evaluation (ROUGE)", "Word Error Rate (WER)", "Bilingual Evaluation Understudy (BLEU)"],
  //   correctAnswers: ["Recall-Oriented Understudy for Gisting Evaluation (ROUGE)"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "An AI customer service agent, is unable to accurately identify Customer Intent based on Customer Message. You can improve it's performance by using training data in which format:",
  //   options: ["Customer Message and Customer Intent", "Customer Message and Agent Resposne", "Customer Intent and Agent Resposne", "Agent Response and Customer Intent"],
  //   correctAnswers: ["Customer Message and Customer Intent"],
  //   multipleAllowed: false
  // },   
  // {
  //   question: "How are users typically charged for using a foundation model? [Select Two]",
  //   options: ["Number of Input Tokens", "Number of Output Tokens", "Model Architecture", "Inference Latency"],
  //   correctAnswers: ["Number of Input Tokens", "Number of Output Tokens"],
  //   multipleAllowed: true,
  //   maxSelections: 2
  // },   
  // {
  //   question: "Which AWS AI Service can be used to extract health data from unstructured text such as clinical notes & medical records?",
  //   options: ["Amazon Comprehend Medical", "Amazon Transcribe Medical", "Amazon HealthLake", "Amazon Rekognition"],
  //   correctAnswers: ["Amazon Comprehend Medical"],
  //   multipleAllowed: false
  // },   
  // {
  //   question: "Which type of machine learning model is specifically designed to generate new data that resembles existing data?",
  //   options: ["Autoencoder", "Generative Adversarial Network (GAN)", "Decision Tree", "Support Vector Machine (SVM)"],
  //   correctAnswers: ["Generative Adversarial Network (GAN)"],
  //   multipleAllowed: false
  // },             
  // {
  //   question: "Users are going to use long prompts to ask questions from their Large Language Model (LLM). What key aspect should be considered while selecting the LLM to use?",
  //   options: ["Inference Latency", "Maximum Context Window", "Model Size", "Training Data"],
  //   correctAnswers: ["Maximum Context Window"],
  //   multipleAllowed: false
  // },  
  // {
  //   question: "Which of the following best describes the primary purpose of Amazon SageMaker Feature Store?",
  //   options: ["To automatically train and deploy machine learning models", "To store and manage features for machine learning workflow", "To provide a marketplace for pre-trained machine learning models", "To optimize the performance of SageMaker training jobs"],
  //   correctAnswers: ["To store and manage features for machine learning workflow"],
  //   multipleAllowed: false
  // },  
  // {
  //   question: "A healthcare organization is developing an AI-powered diagnostic tool to assist in early detection of a rare disease. With respect to regulatory compliance concerns - which of the following is least relevant?",
  //   options: ["Ensuring the AI system is unbiased and doesn ot discriminate against certain patient demographics", "Minimizing operational expesese of the AI system", "Ensuring the AI system is transparent in it's decision-making process", "Preventing the AI system from being used for unauthorized purposes"],
  //   correctAnswers: ["Minimizing operational expesese of the AI system"],
  //   multipleAllowed: false
  // },  
  // {
  //   question: "You're a large enterprise with a massive amount of unstructured data scattered across various internal systems. You want to provide your employees with a powerful search tool that can understand natural language queries and return accurate, relevant results. Which AWS service would best meet this need?",
  //   options: ["Amazon RedShift", "Amazon Lex", "Amazon Kendra", "Amazon DynamoDB"],
  //   correctAnswers: ["Amazon Kendra"],
  //   multipleAllowed: false
  // },  
  // {
  //   question: "A data scientist is working on a project that requires rapid prototyping and experimentation with various machine learning algorithms. Which AWS service would be most suitable for this task?",
  //   options: ["Amazon SageMaker Ground Truth", "Amazon Elastic Compute Cloud (EC2)", "Amazon SageMaker Autopilot", "Amazon Bedrock"],
  //   correctAnswers: ["Amazon SageMaker Autopilot"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A large company wants to create an application for their Sales Managers - that can reason, perform multi-step tasks and provide insightful responses from their enterprise data. Which AWS service would be most suitable for this task?",
  //   options: ["Amazon Lex", "Amazon SageMaker", "Amazon Bedrock Knowledgebases", "Amazon Bedrock Agents"],
  //   correctAnswers: ["Amazon Bedrock Agents"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A company wants to analyze customer reviews to identify common themes and sentiments. Which AWS service can the company use to meet this requirements?",
  //   options: ["Amazon Connect", "Amazon Comprehend", "Amazon Translate", "Amazon Transcribe"],
  //   correctAnswers: ["Amazon Comprehend"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A company wants to transform data from one format to another to prepare it for machine learning tasks. Which AWS service is best suited for this data transformation?",
  //   options: ["AWS Glue", "Amazon Translate", "AWS config", "Amazon Kinesis"],
  //   correctAnswers: ["AWS Glue"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A company wants to deploy a trained machine learning model for real-time inference. Which AWS service would be most suitable for this purpose?",
  //   options: ["Amazon SageMaker", "Amazon Personalize", "Amazon Elastic Compute Cloud (EC2)", "Amazon SageMaker Endpoints"],
  //   correctAnswers: ["Amazon SageMaker Endpoints"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A company has deployed a machine learning model for customer sentiment analysis. To ensure the model's accuracy and reliability, which AWS services should be used for monitoring and human review? [Select Two]",
  //   options: ["Amazon Bedrock", "Amazon SageMaker Model Monitor", "Amazon SageMaker Ground Truth", "Amazon A2I (Amazon Augmented AI)"],
  //   correctAnswers: ["Amazon SageMaker Model Monitor", "Amazon A2I (Amazon Augmented AI)"],
  //   multipleAllowed: true,
  //   maxSelections: 2
  // },
  // {
  //   question: "A ML specialist is training a large deep learning model on a massive dataset in Amazon SageMaker - a single GPU may not handle this well. Which SageMaker feature can help optimize the training process for large models and datasets?",
  //   options: ["Incremental Training", "Hyperparameter tuning", "Pipe Mode", "Model Parallelism"],
  //   correctAnswers: ["Model Parallelism"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "You're working with a large dataset with many features. To improve your model's performance and computational efficiency, you need to simplify the data without losing significant information. Wihch technique would be most effective for achieving this goal?",
  //   options: ["Dimensionality Reduction", "Feature Engineering", "Data Augmentation", "Data Cleaning"],
  //   correctAnswers: ["Dimensionality Reduction"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "You want to generate highly detailed images based on text descriptions. Which AI model, specifically designed for generative tasks and capable of producing high-quality, diverse outputs, would be most suitable for this task?",
  //   options: ["Generative Adversarial Networks (GANs)", "Recurrent Neural Networks (RNNS)", "Convolutional Neural Networks (CNNs)", "Stable Diffusion"],
  //   correctAnswers: ["Stable Diffusion"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A company has a system that generates vector embeddings from product data. They want to improve the speed and accuracy of finding similar products. Which AWS services are best suited for implementing vector search to optimize the system? [Select Three]",
  //   options: ["Amazon OpenSearch Service", "Amazon RedShift", "Amazon Neptune", "Amazon DocumentDB (with MongoDB compatibility)"],
  //   correctAnswers: ["Amazon OpenSearch Service", "Amazon Neptune", "Amazon DocumentDB (with MongoDB compatibility)"],
  //   multipleAllowed: true,
  //   maxSelections: 3
  // },
  // {
  //   question: "A bank receives numerous load applications daily. The load processing team manually extracts information from these applications, which is time-consuming. The goal is to automate this process using AI tools. Which AWS service would be useful here?",
  //   options: ["Amazon Rekognition", "Amazon Textract", "Amazon Translate", "Amazon Transcribe"],
  //   correctAnswers: ["Amazon Textract"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "A healthcare company wants to develop a ML model to predict the likelihood of a patient developing diabetes based on various health indicators. Which of the following metrics would be most appropriate for evaluating the model's performance? [Select Two]",
  //   options: ["Accuracy", "Precision", "F1-Score", "Recall (Sensitivity)", "Area Under ROC Curve (AUC-ROC"],
  //   correctAnswers: ["Recall (Sensitivity)", "Area Under ROC Curve (AUC-ROC"],
  //   multipleAllowed: true,
  //   maxSelections: 2
  // },
  // {
  //   question: "An organization has trained a deep learning model on a large dataset of general images. They now wnat to apply the same model to classify medical images with a smaller (additional training) dataset. Which machine learning techique would be most suitable in this scenario?",
  //   options: ["Reinforcement Learning", "Transfer Learning", "Supervised Learning", "Unsupervised Learning"],
  //   correctAnswers: ["Transfer Learning"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "You are building a machine learning model on AWS and want to share it securely with a third-party partner. Which AWS service would you use to establish a private connection between your VPC and the partner's VPC, ensuring that the data remains within your AWS account and is not exposed to the public internet?",
  //   options: ["AWS Direct Connect", "AWS PrivateLink", "AWS Transit Gateway", "AWS VPN"],
  //   correctAnswers: ["AWS PrivateLink"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "You are training on machine learning model on sensitive customer data using AWS SageMaker. Under the AWS Shared Responsibility model, which of the following is primarily your responsibility?",
  //   options: ["Securing the AWS SageMaker infrastructure", "Protecting underlying operating system of the SageMaker instance", "Ensuring security for customer data stored in S3", "Patching the AWS SageMaker software"],
  //   correctAnswers: ["Ensuring security for customer data stored in S3"],
  //   multipleAllowed: false
  // },
  // {
  //   question: "When implementing the Generative AI Security Scoping Matrix, which of the following factors should be assesed to determine the level of risk associated with a generative AI project?",
  //   options: ["The model's computational efficiency", "The sentivity of the data used to train the model", "Inference Latency", "The number of parameters in the model"],
  //   correctAnswers: ["The sentivity of the data used to train the model"],
  //   multipleAllowed: false
  // },
]

const optionLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
const TOTAL_SCORE = 1000
const PASS_THRESHOLD = 700
const POINTS_PER_QUESTION = TOTAL_SCORE / questions.length

export default function Component() {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [score, setScore] = useState(0)
  const [selectedAnswers, setSelectedAnswers] = useState<string[]>([])
  const [quizCompleted, setQuizCompleted] = useState(false)
  const [userAnswers, setUserAnswers] = useState<string[][]>([])
  const [passed, setPassed] = useState(false)
  const [passPercentage, setPassPercentage] = useState(0)

  const handleSelectOption = (option: string) => {
    const currentQuestionData = questions[currentQuestion]
    if (currentQuestionData.multipleAllowed) {
      if (selectedAnswers.includes(option)) {
        setSelectedAnswers(prev => prev.filter(a => a !== option))
      } else if (selectedAnswers.length < (currentQuestionData.maxSelections || Infinity)) {
        setSelectedAnswers(prev => [...prev, option])
      }
    } else {
      setSelectedAnswers([option])
    }
  }

  const handleNextQuestion = () => {
    const currentQuestionData = questions[currentQuestion]
    const isCorrect = currentQuestionData.multipleAllowed
      ? currentQuestionData.correctAnswers.every(a => selectedAnswers.includes(a)) &&
        selectedAnswers.length === currentQuestionData.correctAnswers.length
      : selectedAnswers[0] === currentQuestionData.correctAnswers[0]

    if (isCorrect) {
      setScore(score + POINTS_PER_QUESTION)
    }

    setUserAnswers([...userAnswers, selectedAnswers])

    if (currentQuestion + 1 < questions.length) {
      setCurrentQuestion(currentQuestion + 1)
      setSelectedAnswers([])
    } else {
      const finalScore = score + (isCorrect ? POINTS_PER_QUESTION : 0)
      const hasPassed = finalScore >= PASS_THRESHOLD
      const percentage = (finalScore / TOTAL_SCORE) * 100
      setPassed(hasPassed)
      setPassPercentage(Math.round(percentage))
      setQuizCompleted(true)
      if (hasPassed) {
        setTimeout(() => {
          confetti({
            particleCount: 100,
            spread: 70,
            origin: { y: 0.6 }
          })
        }, 500)
      }
    }
  }

  const handleResetQuiz = () => {
    setCurrentQuestion(0)
    setScore(0)
    setSelectedAnswers([])
    setQuizCompleted(false)
    setUserAnswers([])
    setPassed(false)
    setPassPercentage(0)
  }

  const currentQuestionData = questions[currentQuestion]
  const selectionLimitReached = currentQuestionData.multipleAllowed && 
    selectedAnswers.length >= (currentQuestionData.maxSelections || Infinity)

  const isNextButtonDisabled = currentQuestionData.multipleAllowed
    ? selectedAnswers.length !== currentQuestionData.maxSelections
    : selectedAnswers.length === 0

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 text-gray-100 p-4 bg-pattern">
      <style jsx global>{`
        .bg-pattern {
          background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
        }
      `}</style>
      <div className="w-full max-w-2xl bg-gray-800 bg-opacity-80 backdrop-blur-lg rounded-xl shadow-lg p-4 sm:p-6 md:p-8">
        <div className="flex justify-center mb-6">
          <Image
            src="/logo.png"
            alt="Quiz Logo"
            width={180}
            height={180}
            className="rounded-full bg-gray-700 p-2"
          />
        </div>
        <AnimatePresence mode="wait">
          {!quizCompleted ? (
            <motion.div
              key="question"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
              <div className="mb-4 text-lg font-semibold text-blue-300">
                Question {currentQuestion + 1} of {questions.length}
              </div>
              <h2 className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6 text-gray-100">
                {currentQuestionData.question}
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 mb-4 sm:mb-6">
                {currentQuestionData.options.map((option, index) => (
                  <button
                    key={option}
                    onClick={() => handleSelectOption(option)}
                    disabled={selectionLimitReached && !selectedAnswers.includes(option)}
                    className={`p-3 sm:p-4 rounded-lg text-left transition-all duration-300 ${
                      selectedAnswers.includes(option)
                        ? 'bg-green-600 text-white'
                        : selectionLimitReached
                        ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                        : 'bg-gray-700 hover:bg-gray-600 text-gray-100'
                    }`}
                  >
                    <span className="font-bold mr-2">{optionLetters[index]}.</span> {option}
                  </button>
                ))}
              </div>
              {currentQuestionData.multipleAllowed && (
                <p className="text-sm text-blue-300 mb-4">
                  This question requires {currentQuestionData.maxSelections} selections. 
                  You have selected {selectedAnswers.length} out of {currentQuestionData.maxSelections}.
                </p>
              )}
              <button
                onClick={handleNextQuestion}
                disabled={isNextButtonDisabled}
                className={`w-full p-3 sm:p-4 rounded-lg text-lg font-semibold transition-all duration-300 ${
                  !isNextButtonDisabled
                    ? 'bg-orange-500 hover:bg-orange-600 text-white'
                    : 'bg-gray-600 cursor-not-allowed text-gray-400'
                }`}
              >
                {currentQuestion === questions.length - 1 ? 'Finish Quiz' : 'Next Question'}
              </button>
            </motion.div>
          ) : (
            <motion.div
              key="results"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className="text-center"
            >
              <h2 className="text-2xl sm:text-3xl font-bold mb-4 sm:mb-6 text-gray-100">Quiz Completed!</h2>
              <p className="text-xl sm:text-2xl mb-4 sm:mb-6 text-blue-300">
                Your score: {Math.round(score)} / {TOTAL_SCORE}
              </p>
              <p className={`text-xl sm:text-2xl mb-2 ${passed ? 'text-green-400' : 'text-red-400'}`}>
                {passed 
                  ? `Congratulations! You passed with ${passPercentage}%` 
                  : `Sorry, you did not pass. You scored ${passPercentage}%`}
              </p>
              <div className="mt-6 sm:mt-8 text-left">
                <h3 className="text-lg sm:text-xl font-semibold mb-4 text-gray-100">Question Summary:</h3>
                {questions.map((q, index) => (
                  <div key={index} className="mb-4 sm:mb-6 bg-gray-700 bg-opacity-50 rounded-lg p-3 sm:p-4">
                    <p className="font-medium mb-2 text-gray-100">
                      {index + 1}. {q.question}
                    </p>
                    <div className="mb-2">
                      {q.options.map((option, optionIndex) => (
                        <p key={optionIndex} className={`${q.correctAnswers.includes(option) ? 'text-green-400' : 'text-gray-300'}`}>
                          {optionLetters[optionIndex]}. {option}
                        </p>
                      ))}
                    </div>
                    <p className={`flex items-center ${
                      JSON.stringify(userAnswers[index].sort()) === JSON.stringify(q.correctAnswers.sort()) 
                        ? 'text-green-400' 
                        : 'text-red-400'
                    }`}>
                      Your Answer(s): {userAnswers[index].map(answer => optionLetters[q.options.indexOf(answer)]).join(", ")}
                      {JSON.stringify(userAnswers[index].sort()) === JSON.stringify(q.correctAnswers.sort()) ? (
                        <CheckCircle className="ml-2 w-5 h-5" />
                      ) : (
                        <XCircle className="ml-2 w-5 h-5" />
                      )}
                    </p>
                  </div>
                ))}
              </div>
              <button
                onClick={handleResetQuiz}
                className="mt-4 sm:mt-6 p-3 sm:p-4 bg-orange-500 hover:bg-orange-600 rounded-lg text-lg font-semibold transition-all duration-300 text-white w-full"
              >
                Reset Quiz
              </button>
            </motion.div>
          )}
        </AnimatePresence>
        <div className="mt-6 sm:mt-8 text-center text-gray-400 text-sm">
          Made by <a target='_blank' href='https://linkedin.com/in/ptuladhar3'>Puru Tuladhar</a>
        </div>
      </div>
    </div>
  )
}