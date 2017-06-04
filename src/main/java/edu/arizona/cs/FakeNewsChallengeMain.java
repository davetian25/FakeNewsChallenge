package edu.arizona.cs;

//Standard Java imports for File I/O and utilities
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;

// Imports from Apache Lucene, for tf-idf weighting and other analyzing
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;

// Import for the CSV file reader from opencsv
import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
// Import from StanfordNLP for Sentence structs
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.util.CoreMap;
//Imports for the Weka ML algorithms
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class FakeNewsChallengeMain {

	// HashMap of the body of articles
	static HashMap<String, MyDocument> articles = new HashMap<String, MyDocument>();
	// List of all headlines in the training set
	static List<Headline> trainingHeadlines = new ArrayList<Headline>();
	// List of all the headlines in the testing set
	static List<Headline> testingHeadlines = new ArrayList<Headline>();
	// List of the results of testing
	static List<Headline> testingResults = new ArrayList<Headline>();
	
	// HashMap of official data documents
	static HashMap<String, MyDocument> officialArticles = new HashMap<String, MyDocument>();
	// List of official headline stances
	static List<Headline> officialHeadlines = new ArrayList<Headline>();
	// List of the official results of testing
	static List<Headline> officialResults = new ArrayList<Headline>();


	/** The counters for prior probabilities **/
	static float headcount = 0;
	static float artcount = 0;
	static float unrelatedHeadlines = 0;
	static float agreesHeadlines = 0;
	static float disagreesHeadlines = 0;
	static float discussesHeadlines = 0;

	// Punctuation marks to be ignored when building a query -- lexical errors if included in a query
	static String punctuations = ".,:;\"\'`!/?-";
	// List of words that tend to show agreement
	static List<String> agreeWords = Arrays.asList("explains", "explain", "insists", "insist", "confirms");
	// List of words that tend to show disagreement
	static List<String> disagreeWords = Arrays.asList("not", "NOT", "doubt", "casts");
	// List of words that tend to show a discussion
	static List<String> discussWords = Arrays.asList("alleged", "allegedly", "reportedly", "claiming", "claims",
			"claim", "investigate", "may", "could", "explain", "appear", "speculation", "checks", "is");
	// List of words that trip up the Lucene QueryParser
	static List<String> tripWords = Arrays.asList("-lrb-", "-rrb-", "NOT", "???", "!?", "--", "``", "??");
	// List of words that are to be split because of lexical issues
	static List<String> splitWords = Arrays.asList("9/11-style", "15/month", "9/11", "Code/Red", "PCs/printers", "pcs/printers", "PC/printer", "pc/printer");
	// List of words with a generally positive connotation
	static List<String> posWords = Arrays.asList("confirms", "confirm", "born");
	// List of words with a generally negative connotation
	static List<String> negWords = Arrays.asList("beheaded", "behead", "bail", "bails", "quits", "won't", "wont", "not",
			"intercept", "intercepted", "shot", "bombed", "bomb", "explosion", "explode");

	/** Global variable for the Weka Naive Bayes implementation **/
	// The collection of Instances belonging to the training set
	static Instances training_set;
	// The collection of Instances belonging to the testing set
	static Instances testing_set;
	// The collection of Instances belonging to the official testing set
	static Instances official_set;
	// The classification model, specified as Naive Bayes
	static Classifier cModel;

	static int numberOfHeadlines;

	public static void main(String[] args) throws IOException, ParseException {
		
		
		/****************************************
		 * The first half - designing the model *
		 ****************************************/
		
		String path_to_training_bodies, 
		       path_to_training_stances, 
		       path_to_testing_stances;
		path_to_training_bodies = getRelativePath(args[0]);
	    path_to_training_stances = getRelativePath(args[4]);
	    path_to_testing_stances = getRelativePath(args[5]);
		/*** Use the opencsv jar for reading the csv file ***/
		// Read the bodies of the training data
		System.out.println("Preprocessing the " + args[0] + " file");
		CSVReader reader_bodies_train = new CSVReader(new FileReader(path_to_training_bodies));
		String[] nextLine_bodies_train;
		while ((nextLine_bodies_train = reader_bodies_train.readNext()) != null) {
			String bodyID = nextLine_bodies_train[0];
			String article = nextLine_bodies_train[1];
			createTheTrainDocument(bodyID, article);
		}
		reader_bodies_train.close();
		
		// Read the headlines and stances of the training data
		CSVReader reader_stances_training = new CSVReader(new FileReader(path_to_training_stances));
		String[] nextLine_stances_training;
		System.out.println("Preprocessing the " + args[4] + " file");
		while ((nextLine_stances_training = reader_stances_training.readNext()) != null) {
			String headline = nextLine_stances_training[0];
			String bodyID = nextLine_stances_training[1];
			String actualStance = nextLine_stances_training[2];
			createTheTrainHeadline(headline, bodyID, actualStance);
		}
		reader_stances_training.close();
		
		// Preprocessing for Naive Bayes Training
		preProcessForNBTraining(1);
		System.out.println("Beginning training for preprocessed data");
		try {
			makeInstances(1);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
		System.out.println("Generating an .arff file for Weka");
		generateArffFile("test_stances_csc483583.arff");
		
		// Read the headlines and stances of the testing data
		CSVReader tester_stances_pretest = new CSVReader(new FileReader(path_to_testing_stances));
		String[] tester_nextLine_stances_pretest;
		System.out.println("Preprocessing the " + args[5] + " file");
		while ((tester_nextLine_stances_pretest = tester_stances_pretest.readNext()) != null) {
			String headline = tester_nextLine_stances_pretest[0];
			String bodyID = tester_nextLine_stances_pretest[1];
			String actualStance = tester_nextLine_stances_pretest[2];
			createTheTestHeadline(headline, bodyID, actualStance);
		}
		tester_stances_pretest.close();
		
		// Preprocessing for Naive Bayes Testing
		preProcessforNBTesting(1);
		System.out.println("Beginning testing for preprocessed data");
		try {
			applyClassifyingModel(1);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("Generating the output file from phase one, and computing the performance score");
		generateOutputCSV("phase_one_testing.csv", 1);
		getFinalScore(1);
		

		/**************************************
		 * The second half - official testing *
		 **************************************/

		// Strings for the relative paths to all three of the files
		String official_test_bodies, official_test_stances;
		official_test_bodies = getRelativePath(args[2]);
		official_test_stances = getRelativePath(args[3]);

		/*** Use the opencsv jar for reading the csv file ***/
		// Read the bodies of the official data
		System.out.println("Preprocessing the " + args[2] + " file");
		CSVReader official_bodies = new CSVReader(new FileReader(official_test_bodies));
		String[] official_nextLine_bodies;
		while ((official_nextLine_bodies = official_bodies.readNext()) != null) {
			String bodyID = official_nextLine_bodies[0];
			String article = official_nextLine_bodies[1];
			createTheOfficialDocument(bodyID, article);
		}
		official_bodies.close();
		
		/*** Use the opencsv jar for reading the csv file ***/
		// Read the stances of the official data
		System.out.println("Preprocessing the " + args[3] + " file");
		CSVReader reader_stances_official = new CSVReader(new FileReader(official_test_stances));
		String[] nextLine_stances_official;
		while ((nextLine_stances_official = reader_stances_official.readNext()) != null) {
			String headline = nextLine_stances_official[0];
			String bodyID = nextLine_stances_official[1];
			createTheOfficialHeadline(headline, bodyID);
		}
		reader_stances_official.close();
		
		preProcessforNBTesting(2);
		
		System.out.println("Beginning Naive Bayes testing for preprocessed data");
		try {
			applyClassifyingModel(2);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		generateOutputCSV("phase_two_testing.csv", 2);
		getFinalScore(2);

	}

	/**
	 * The creation and instantiation of the Instances that are added to the
	 * training data set before a classifier is derived from the values that are
	 * being used
	 **/
	public static void makeInstances(int phase) throws Exception {

		// Declare the first Attribute, the tf_idf
		// Attribute Attribute1 = new Attribute("tf-idf");
		// Declare second Attribute, whether it is agreeable or not
		Attribute Attribute2 = new Attribute("agree");
		// Declare third Attribute, whether it is disagreeable or not
		Attribute Attribute3 = new Attribute("disagree");
		// Declare fourth Attribute, whether it is discussive or not
		Attribute Attribute4 = new Attribute("discuss");
		// Declare fifth Attribute, whether the sentiment is positive or not
		Attribute Attribute5 = new Attribute("positive");
		// Declare sixth attribute, sentiment analysis
		Attribute Attribute6 = new Attribute("sentiment");

		// Declare the class attribute along with its values
		ArrayList<String> fvClassVal = new ArrayList<String>();
		fvClassVal.add("unrelated");
		fvClassVal.add("agree");
		fvClassVal.add("disagree");
		fvClassVal.add("discuss");
		Attribute ClassAttribute = new Attribute("the_class", fvClassVal);

		// Declare the feature vector
		ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>();
		// fvWekaAttributes.add(Attribute1);
		fvWekaAttributes.add(Attribute2);
		fvWekaAttributes.add(Attribute3);
		fvWekaAttributes.add(Attribute4);
		fvWekaAttributes.add(Attribute5);
		fvWekaAttributes.add(Attribute6);
		fvWekaAttributes.add(ClassAttribute);

		// Create an empty training set
		training_set = new Instances("Train Data", fvWekaAttributes, 10000);
		// Set class index
		training_set.setClassIndex(fvWekaAttributes.size() - 1);

		for (Headline headline : trainingHeadlines) {
			if (!headline.getBodyID().equals("Body ID")) {
				if (headline.related) {
					// Create the instance
					Instance iExample = new DenseInstance(6);
					// iExample.setValue((Attribute) fvWekaAttributes.get(0), headline.getTFIDF());
					iExample.setValue((Attribute) fvWekaAttributes.get(0), headline.agrees);
					iExample.setValue((Attribute) fvWekaAttributes.get(1), headline.disagrees);
					iExample.setValue((Attribute) fvWekaAttributes.get(2), headline.discusses);
					iExample.setValue((Attribute) fvWekaAttributes.get(3), headline.isPositive());
					iExample.setValue((Attribute) fvWekaAttributes.get(4), headline.sentiment);
					String headlineStance = headline.actualStance;
					iExample.setValue((Attribute) fvWekaAttributes.get(5), headlineStance);
					// add the instance
					training_set.add(iExample);
				}
			}
		}
		// Create a naïve bayes classifier
		cModel = (Classifier) new NaiveBayes();
		cModel.buildClassifier(training_set);
		System.out.println("The model is:");
		System.out.println(cModel);

	}

	/**
	 * The application of the Weka Naive Bayes Machine Learning algorithm, using
	 * several distinct attributes for supervised learning
	 **/
	public static void applyClassifyingModel(int phase) throws Exception {

		if(phase == 1) {
			// Declare the first Attribute, the tf_idf
			// Attribute Attribute1 = new Attribute("tf-idf");
			// Declare second Attribute, whether it is agreeable or not
			Attribute Attribute2 = new Attribute("agree");
			// Declare third Attribute, whether it is disagreeable or not
			Attribute Attribute3 = new Attribute("disagree");
			// Declare fourth Attribute, whether it is discussive or not
			Attribute Attribute4 = new Attribute("discuss");
			// Declare fifth Attribute, whether the sentiment is positive or not
			Attribute Attribute5 = new Attribute("positive");
			// Declare sixth attribute, sentiment analysis
			Attribute Attribute6 = new Attribute("sentiment");

			// Declare the class attribute along with its values
			ArrayList<String> fvClassVal = new ArrayList<String>();
			fvClassVal.add("unrelated");
			fvClassVal.add("agree");
			fvClassVal.add("disagree");
			fvClassVal.add("discuss");
			Attribute ClassAttribute = new Attribute("the_class", fvClassVal);

			// Declare the feature vector
			ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>();
			// fvWekaAttributes.add(Attribute1);
			fvWekaAttributes.add(Attribute2);
			fvWekaAttributes.add(Attribute3);
			fvWekaAttributes.add(Attribute4);
			fvWekaAttributes.add(Attribute5);
			fvWekaAttributes.add(Attribute6);
			fvWekaAttributes.add(ClassAttribute);

			// Create an empty training set
			testing_set = new Instances("Train Data", fvWekaAttributes, 10000);
			// Set class index
			testing_set.setClassIndex(fvWekaAttributes.size() - 1);

			ArrayList<Headline> relatedHeadlines = new ArrayList<Headline>();
			
			for (Headline headline : testingHeadlines) {
				if (!headline.getBodyID().equals("Body ID")) {
					if (headline.related) {
						// Keeping track of any mistakes made by the classifier
						relatedHeadlines.add(headline);
						// Create the instance
						Instance iExample = new DenseInstance(6);
						// iExample.setValue((Attribute) fvWekaAttributes.get(0), headline.getTFIDF());
						iExample.setValue((Attribute) fvWekaAttributes.get(0), headline.agrees);
						iExample.setValue((Attribute) fvWekaAttributes.get(1), headline.disagrees);
						iExample.setValue((Attribute) fvWekaAttributes.get(2), headline.discusses);
						iExample.setValue((Attribute) fvWekaAttributes.get(3), headline.isPositive());
						iExample.setValue((Attribute) fvWekaAttributes.get(4), headline.sentiment);
						String headlineStance = headline.actualStance;
						iExample.setValue((Attribute) fvWekaAttributes.get(5), headlineStance);
						// add the instance
						testing_set.add(iExample);
					}
				}
			}
			// Test the model
			Evaluation eTest = new Evaluation(testing_set);
			eTest.evaluateModel(cModel, testing_set);
			
			 ArrayList<Prediction> Predictions = eTest.predictions();
			 int predictionCounter = 0;
			 for (Prediction predict : Predictions) {
				 if (predict.predicted() == predict.actual()) {
					 Headline correctHeadline = relatedHeadlines.get(predictionCounter);
					 correctHeadline.correctlyClassed=true;
					 correctHeadline.setCorrectlyClassified();
					 correctHeadline.predictedClass = predict.predicted();
					 testingResults.add(correctHeadline);
				 } else {
					 Headline correctHeadline = relatedHeadlines.get(predictionCounter);
					 correctHeadline.predictedClass = predict.predicted();
					 testingResults.add(correctHeadline);
				 }
				 predictionCounter++;
			 }

			String classDeets = eTest.toClassDetailsString();
			System.out.println("Class Details " + classDeets);

			// Print the result à la Weka explorer:
			String strSummary = eTest.toSummaryString();
			System.out.println("Performance Summary " + strSummary);
		} else {
			// Declare the first Attribute, the tf_idf
			// Attribute Attribute1 = new Attribute("tf-idf");
			// Declare second Attribute, whether it is agreeable or not
			Attribute Attribute2 = new Attribute("agree");
			// Declare third Attribute, whether it is disagreeable or not
			Attribute Attribute3 = new Attribute("disagree");
			// Declare fourth Attribute, whether it is discussive or not
			Attribute Attribute4 = new Attribute("discuss");
			// Declare fifth Attribute, whether the sentiment is positive or not
			Attribute Attribute5 = new Attribute("positive");
			// Declare sixth attribute, sentiment analysis
			Attribute Attribute6 = new Attribute("sentiment");

			// Declare the class attribute along with its values
			ArrayList<String> fvClassVal = new ArrayList<String>();
			fvClassVal.add("unrelated");
			fvClassVal.add("agree");
			fvClassVal.add("disagree");
			fvClassVal.add("discuss");
			Attribute ClassAttribute = new Attribute("the_class", fvClassVal);

			// Declare the feature vector
			ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>();
			// fvWekaAttributes.add(Attribute1);
			fvWekaAttributes.add(Attribute2);
			fvWekaAttributes.add(Attribute3);
			fvWekaAttributes.add(Attribute4);
			fvWekaAttributes.add(Attribute5);
			fvWekaAttributes.add(Attribute6);
			fvWekaAttributes.add(ClassAttribute);

			// Create an empty training set
			official_set = new Instances("Train Data", fvWekaAttributes, 10000);
			// Set class index
			official_set.setClassIndex(fvWekaAttributes.size() - 1);

			ArrayList<Headline> relatedHeadlines = new ArrayList<Headline>();
			
			for (Headline headline : officialHeadlines) {
				if (!headline.getBodyID().equals("Body ID")) {
					if (headline.related) {
						// Keeping track of any mistakes made by the classifier
						relatedHeadlines.add(headline);
						// Create the instance
						Instance iExample = new DenseInstance(6);
						// iExample.setValue((Attribute) fvWekaAttributes.get(0), headline.getTFIDF());
						iExample.setValue((Attribute) fvWekaAttributes.get(0), headline.agrees);
						iExample.setValue((Attribute) fvWekaAttributes.get(1), headline.disagrees);
						iExample.setValue((Attribute) fvWekaAttributes.get(2), headline.discusses);
						iExample.setValue((Attribute) fvWekaAttributes.get(3), headline.isPositive());
						iExample.setValue((Attribute) fvWekaAttributes.get(4), headline.sentiment);
						String headlineStance = headline.actualStance;
						iExample.setValue((Attribute) fvWekaAttributes.get(5), headlineStance);
						// add the instance
						official_set.add(iExample);
					}
				}
			}
			// Test the model
			Evaluation eTest = new Evaluation(official_set);
			eTest.evaluateModel(cModel, official_set);
			
			 ArrayList<Prediction> Predictions = eTest.predictions();
			 int predictionCounter = 0;
			 for (Prediction predict : Predictions) {
				 if (predict.predicted() == predict.actual()) {
					 Headline correctHeadline = relatedHeadlines.get(predictionCounter);
					 correctHeadline.correctlyClassed=true;
					 correctHeadline.setCorrectlyClassified();
					 correctHeadline.predictedClass = predict.predicted();
					 officialResults.add(correctHeadline);
				 } else {
					 Headline correctHeadline = relatedHeadlines.get(predictionCounter);
					 correctHeadline.predictedClass = predict.predicted();
					 officialResults.add(correctHeadline);
				 }
				 predictionCounter++;
			 }

			String classDeets = eTest.toClassDetailsString();
			System.out.println("Class Details " + classDeets);

			// Print the result à la Weka explorer:
			String strSummary = eTest.toSummaryString();
			System.out.println("Performance Summary " + strSummary);
		}
		

	}

	/**
	 * Used to generate a .arff file, which is a unique file extension used
	 * extensively by the Weka ML Library
	 **/
	public static void generateArffFile(String filename) {

		BufferedWriter bw = null;
		FileWriter fw = null;
		int loadScreen = Math.round(trainingHeadlines.size() / 75);
		int counter = 0;

		try {
			String header = "@RELATION First\n\n@ATTRIBUTE tf_idf NUMERIC\n@ATTRIBUTE related {true,false}\n\n@data\n";

			fw = new FileWriter(filename);
			bw = new BufferedWriter(fw);
			bw.write(header);
			for (Headline headline : trainingHeadlines) {
				if (counter % loadScreen == 0) {
					System.out.print("#");
				}
				bw.write(headline.generateArffEntry() + "\n");
				counter++;
			}
			System.out.println();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (bw != null)
					bw.close();
				if (fw != null)
					fw.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}

	}
	
	/**
	 * Used to preclassify any of the headlines that are determined to be 
	 * unrelated, using the tf-idf score
	 **/
	public static void preProcessforNBTesting(int phase) throws IOException, ParseException {
		if(phase == 1) {
			// Preprocess for each headline in the list
			for (Headline headline : testingHeadlines) {
				// Calculate the different scores for the four classes
				MyDocument mappedDoc = articles.get(headline.getBodyID());
				float tf_idf = (float) 0.0;
				StandardAnalyzer analyzer = new StandardAnalyzer();
				Directory index = new RAMDirectory();
				IndexWriterConfig config = new IndexWriterConfig(analyzer);
				IndexWriter w = new IndexWriter(index, config);
				addDoc(w, mappedDoc.getBodyIDString(), mappedDoc.getArticleString());
				String query = "";
				int termCounter = 0;

				headline.sentiment = mappedDoc.getBodySentiment();

				for (String term : headline.getHeadlineStrings()) {
					if (punctuations.contains(term)) {
						// Do nothing, it'll break the Lucene QueryParser
					} else {
						if (tripWords.contains(term)) {
							continue;
						} else if (splitWords.contains(term)) {
							String[] parts = term.split("-|/");
							if (termCounter == 0) {
							} else {
							}
						} else if (termCounter == 0) {
							query += term;
							termCounter++;
						} else {
							query += " AND " + term;
							termCounter++;
						}
					}
					if (discussWords.contains(term)) {
						// Add points that signify discussion
						headline.addDiscussPoints();
					}
					if (agreeWords.contains(term)) {
						// Add points that signify agreement
						headline.addAgreePoints();
					}
					if (disagreeWords.contains(term)) {
						// Add points that signify disagreement
						headline.addDisagreePoints();
					}
				}

				Query q = new QueryParser("text", analyzer).parse(query);
				int hits = 10;
				w.close();
				IndexReader reader = DirectoryReader.open(index);
				IndexSearcher searcher = new IndexSearcher(reader);
				TopDocs docs = searcher.search(q, hits);
				ScoreDoc[] topHits = docs.scoreDocs;

				if (topHits.length != 0) {
					headline.setTFIDF(topHits[0].score);
					headline.setRelatedness(true);
				} else {
					// QueryParser tf-idf used for ML algorithms
					headline.setTFIDF(tf_idf);
					headline.setRelatedness(false);
					if(headline.actualStance.equals("unrelated")) {
						headline.correctlyClassed=true;
						headline.setCorrectlyClassified();
						headline.realClass=0.0F;
						headline.predictedClass=0.0F;
						testingResults.add(headline);
					}
				}

				// Use the sentiment analysis to set the headline's sentiment
				// If 0.0F, then neutral (discuss),
				// if 1.0F then agree,
				// if 2.0F then disagree
				float headlineSentimentValue = -1.0F;
				float bodySentiment = mappedDoc.getBodySentiment();
				float headlineSentiment = headline.getHeadlineSentiment();
				/*** The values returned by the sentiment getters are the same, 
				 *** as follows:
				 ***  float positive = 1.0F, negative = 2.0F, neutral = 3.0F ****/
				// If the body is neutral and the headline is neutral, they discuss
				// If the body is positive and headline is negative, they disagree -- or vice versa
				// If the body is positive and the headline is positive they agree -- or vice versa
				if (bodySentiment == 1.0F) { // The body is positive
					if (headlineSentiment == 1.0F) { // The headline is positive
						headlineSentimentValue = 1.0F; // They agree
					} else if (headlineSentiment == 2.0F) { // The headline is negative
						headlineSentimentValue = 2.0F; // They disagree
					} else { // The headline is neutral
						headlineSentimentValue = 0.0F; // No determination -- set to neutral
					}
				} else if (bodySentiment == 2.0F) { // The body is negative
					if (headlineSentiment == 1.0F) { // The headline is positive
						headlineSentimentValue = 2.0F; // The disagree
					} else if (headlineSentiment == 2.0F) { // The headline is negative
						headlineSentimentValue = 1.0F; // They agree
					} else { // The headline is neutral
						headlineSentimentValue = 0.0F; // No determination -- set to neutral
					}
				} else if (bodySentiment == 3.0F) { // The body is neutral
					if (headlineSentiment == 1.0F) { // The headline is positive
						headlineSentimentValue = 0.0F; // No determination -- set to neutral
					} else if (headlineSentiment == 2.0F) { // The headline is negative
						headlineSentimentValue = 0.0F; // No determination -- set to neutral
					} else { // The headline is neutral
						headlineSentimentValue = 0.0F; // No determination -- set to neutral
					}
				}

				headline.sentiment = headlineSentimentValue;

				reader.close();

			}
		} else {
			// Preprocess for each headline in the list
			for (Headline headline : officialHeadlines) {
				// Calculate the different scores for the four classes
				MyDocument mappedDoc = officialArticles.get(headline.getBodyID());
				float tf_idf = (float) 0.0;
				StandardAnalyzer analyzer = new StandardAnalyzer();
				Directory index = new RAMDirectory();
				IndexWriterConfig config = new IndexWriterConfig(analyzer);
				IndexWriter w = new IndexWriter(index, config);
				addDoc(w, mappedDoc.getBodyIDString(), mappedDoc.getArticleString());
				String query = "";
				int termCounter = 0;

				headline.sentiment = mappedDoc.getBodySentiment();

				for (String term : headline.getHeadlineStrings()) {
					if (punctuations.contains(term)) {
						// Do nothing, it'll break the Lucene QueryParser
					} else {
						if (tripWords.contains(term)) {
							continue;
						} else if (splitWords.contains(term)) {
							String[] parts = term.split("-|/");
							if (termCounter == 0) {
							} else {
							}
						} else if (termCounter == 0) {
							query += term;
							termCounter++;
						} else {
							query += " AND " + term;
							termCounter++;
						}
					}
					if (discussWords.contains(term)) {
						// Add points that signify discussion
						headline.addDiscussPoints();
					}
					if (agreeWords.contains(term)) {
						// Add points that signify agreement
						headline.addAgreePoints();
					}
					if (disagreeWords.contains(term)) {
						// Add points that signify disagreement
						headline.addDisagreePoints();
					}
				}

				Query q = new QueryParser("text", analyzer).parse(query);
				int hits = 10;
				w.close();
				IndexReader reader = DirectoryReader.open(index);
				IndexSearcher searcher = new IndexSearcher(reader);
				TopDocs docs = searcher.search(q, hits);
				ScoreDoc[] topHits = docs.scoreDocs;

				if (topHits.length != 0) {
					headline.setTFIDF(topHits[0].score);
					headline.setRelatedness(true);
				} else {
					// QueryParser tf-idf used for ML algorithms
					headline.setTFIDF(tf_idf);
					headline.setRelatedness(false);
					if(headline.actualStance.equals("unrelated")) {
						headline.correctlyClassed=true;
						headline.setCorrectlyClassified();
						headline.realClass=0.0F;
						headline.predictedClass=0.0F;
						officialResults.add(headline);
					}
				}

				// Use the sentiment analysis to set the headline's sentiment
				// If 0.0F, then neutral (discuss),
				// if 1.0F then agree,
				// if 2.0F then disagree
				float headlineSentimentValue = -1.0F;
				float bodySentiment = mappedDoc.getBodySentiment();
				float headlineSentiment = headline.getHeadlineSentiment();
				/*** The values returned by the sentiment getters are the same, 
				 *** as follows:
				 ***  float positive = 1.0F, negative = 2.0F, neutral = 3.0F ****/
				// If the body is neutral and the headline is neutral, they discuss
				// If the body is positive and headline is negative, they disagree -- or vice versa
				// If the body is positive and the headline is positive they agree -- or vice versa
				if (bodySentiment == 1.0F) { // The body is positive
					if (headlineSentiment == 1.0F) { // The headline is positive
						headlineSentimentValue = 1.0F; // They agree
					} else if (headlineSentiment == 2.0F) { // The headline is negative
						headlineSentimentValue = 2.0F; // They disagree
					} else { // The headline is neutral
						headlineSentimentValue = 0.0F; // No determination -- set to neutral
					}
				} else if (bodySentiment == 2.0F) { // The body is negative
					if (headlineSentiment == 1.0F) { // The headline is positive
						headlineSentimentValue = 2.0F; // The disagree
					} else if (headlineSentiment == 2.0F) { // The headline is negative
						headlineSentimentValue = 1.0F; // They agree
					} else { // The headline is neutral
						headlineSentimentValue = 0.0F; // No determination -- set to neutral
					}
				} else if (bodySentiment == 3.0F) { // The body is neutral
					if (headlineSentiment == 1.0F) { // The headline is positive
						headlineSentimentValue = 0.0F; // No determination -- set to neutral
					} else if (headlineSentiment == 2.0F) { // The headline is negative
						headlineSentimentValue = 0.0F; // No determination -- set to neutral
					} else { // The headline is neutral
						headlineSentimentValue = 0.0F; // No determination -- set to neutral
					}
				}

				headline.sentiment = headlineSentimentValue;

				reader.close();

			}
		}
		
	}

	/**
	 * Used to determine the values of the selected features which will be used
	 * in the classifying model
	 **/
	public static void preProcessForNBTraining(int phase) throws IOException, ParseException {

		// Preprocess for each headline in the list
		for (Headline headline : trainingHeadlines) {
			// Calculate the different scores for the four classes
			MyDocument mappedDoc = articles.get(headline.getBodyID());
			float tf_idf = (float) 0.0;
			StandardAnalyzer analyzer = new StandardAnalyzer();
			Directory index = new RAMDirectory();
			IndexWriterConfig config = new IndexWriterConfig(analyzer);
			IndexWriter w = new IndexWriter(index, config);
			addDoc(w, mappedDoc.getBodyIDString(), mappedDoc.getArticleString());
			String query = "";
			int termCounter = 0;

			headline.sentiment = mappedDoc.getBodySentiment();

			for (String term : headline.getHeadlineStrings()) {
				if (punctuations.contains(term)) {
					// Do nothing, it'll break the Lucene QueryParser
				} else {
					if (tripWords.contains(term)) {
						continue;
					} else if (splitWords.contains(term)) {
						String[] parts = term.split("-|/");
						if (termCounter == 0) {
						} else {
						}
					} else if (termCounter == 0) {
						query += term;
						termCounter++;
					} else {
						query += " AND " + term;
						termCounter++;
					}
				}
				if (discussWords.contains(term)) {
					// Add points that signify discussion
					headline.addDiscussPoints();
				}
				if (agreeWords.contains(term)) {
					// Add points that signify agreement
					headline.addAgreePoints();
				}
				if (disagreeWords.contains(term)) {
					// Add points that signify disagreement
					headline.addDisagreePoints();
				}
			}

			Query q = new QueryParser("text", analyzer).parse(query);
			int hits = 10;
			w.close();
			IndexReader reader = DirectoryReader.open(index);
			IndexSearcher searcher = new IndexSearcher(reader);
			TopDocs docs = searcher.search(q, hits);
			ScoreDoc[] topHits = docs.scoreDocs;

			if (topHits.length != 0) {
				headline.setTFIDF(topHits[0].score);
				headline.setRelatedness(true);
			} else {
				// QueryParser tf-idf used for ML algorithms
				headline.setTFIDF(tf_idf);
				headline.setRelatedness(false);
				if(headline.actualStance.equals("unrelated")) {
					headline.correctlyClassed=true;
					headline.setCorrectlyClassified();
					headline.predictedClass=0.0F;
				}
			}

			// Use the sentiment analysis to set the headline's sentiment
			// If 0.0F, then neutral (discuss),
			// if 1.0F then agree,
			// if 2.0F then disagree
			float headlineSentimentValue = -1.0F;
			float bodySentiment = mappedDoc.getBodySentiment();
			float headlineSentiment = headline.getHeadlineSentiment();
			/*** The values returned by the sentiment getters are the same, 
			 *** as follows:
			 ***  float positive = 1.0F, negative = 2.0F, neutral = 3.0F ****/
			// If the body is neutral and the headline is neutral, they discuss
			// If the body is positive and headline is negative, they disagree -- or vice versa
			// If the body is positive and the headline is positive they agree -- or vice versa
			if (bodySentiment == 1.0F) { // The body is positive
				if (headlineSentiment == 1.0F) { // The headline is positive
					headlineSentimentValue = 1.0F; // They agree
				} else if (headlineSentiment == 2.0F) { // The headline is negative
					headlineSentimentValue = 2.0F; // They disagree
				} else { // The headline is neutral
					headlineSentimentValue = 0.0F; // No determination -- set to neutral
				}
			} else if (bodySentiment == 2.0F) { // The body is negative
				if (headlineSentiment == 1.0F) { // The headline is positive
					headlineSentimentValue = 2.0F; // The disagree
				} else if (headlineSentiment == 2.0F) { // The headline is negative
					headlineSentimentValue = 1.0F; // They agree
				} else { // The headline is neutral
					headlineSentimentValue = 0.0F; // No determination -- set to neutral
				}
			} else if (bodySentiment == 3.0F) { // The body is neutral
				if (headlineSentiment == 1.0F) { // The headline is positive
					headlineSentimentValue = 0.0F; // No determination -- set to neutral
				} else if (headlineSentiment == 2.0F) { // The headline is negative
					headlineSentimentValue = 0.0F; // No determination -- set to neutral
				} else { // The headline is neutral
					headlineSentimentValue = 0.0F; // No determination -- set to neutral
				}
			}

			headline.sentiment = headlineSentimentValue;

			reader.close();

		}

	}

	/**
	 * Called to create a Document, which is the article associated with a news
	 * headline. It uses a HashMap for speedy lookup
	 **/
	public static void createTheTrainDocument(String bodyID, String theArticle) {
		theArticle = theArticle.replaceAll("\\r\\n|\\r|\\n", " ");
		MyDocument d = null;
		if(!bodyID.equals("Body ID")) {
			if (articles.containsKey(bodyID)) {
				d = (MyDocument) articles.get(bodyID);
			} else {
				d = new MyDocument(theArticle, bodyID);
				articles.put(bodyID, d);
			}	
		}
	}
	public static void createTheOfficialDocument(String bodyID, String theArticle) {
		theArticle = theArticle.replaceAll("\\r\\n|\\r|\\n", " ");
		MyDocument d = null;
		if(!bodyID.equals("Body ID")) {
			if (officialArticles.containsKey(bodyID)) {
				d = (MyDocument) officialArticles.get(bodyID);
			} else {
				d = new MyDocument(theArticle, bodyID);
				officialArticles.put(bodyID, d);
			}		
		}
	}
	public static void createTheOfficialHeadline(String headline, String bodyID) {
		if(!bodyID.equals("Body ID")) {
			Headline h = new Headline(headline, bodyID, "unrelated");
			officialHeadlines.add(h);	
		}
	}

	/**
	 * Called to create a headline for the training headlines List of headlines
	 **/
	public static void createTheTrainHeadline(String headline, String bodyID, String actualStance) {
		if(!bodyID.equals("Body ID")) {
			Headline h = new Headline(headline, bodyID, actualStance);
			trainingHeadlines.add(h);	
		}
	}

	/**
	 * Called to create a headline for the testingHeadlines List of headlines
	 **/
	public static void createTheTestHeadline(String headline, String bodyID, String actualStance) {
		if(!bodyID.equals("Body ID")) {
			Headline h = new Headline(headline, bodyID, actualStance);
			testingHeadlines.add(h);	
		}
	}

	/** Add a document to the collection using Lucene's IndexWriter **/
	private static void addDoc(IndexWriter w, String title, String text) throws IOException {
		Document doc = new Document();
		doc.add(new StringField("title", title, Field.Store.YES));
		doc.add(new TextField("text", text, Field.Store.YES));
		w.addDocument(doc);
	}

	/**
	 * Get the relative path of the three arguments, 
	 * for Weka/Spark/Lucene libraries
	 **/
	public static String getRelativePath(String args) {
		return "./" + args;
	}

	/**
	 * Print out the scoring as it relates to the weighting method specified on
	 * the project web site
	 */
	public static void getFinalScore(int phase) {
		
		if(phase == 1) {
			int[][] confusionMatrix = new int[4][4];
			float correct = 0.0F, incorrect = 0.0F;
			for(Headline headline:testingResults) {
				if(headline.correctlyClassed) {
					correct++;
				} else {
					incorrect++;
				}
				confusionMatrix[(int) headline.realClass][(int) headline.predictedClass]++;
			}
			System.out.println("Correct percentage was " + (float)correct/testingResults.size());
			System.out.println("Incorrect percentage was " + (float)incorrect/testingResults.size());
			System.out.println("\tunrelated | agree | disagree | discuss");
			System.out.println("unrelated: " + String.format("%06d",confusionMatrix[0][0]) + " | " + String.format("%06d",confusionMatrix[0][1]) + " | " + String.format("%06d",confusionMatrix[0][2]) + " | " + String.format("%06d",confusionMatrix[0][3]));
			System.out.println("agree    : " + String.format("%06d",confusionMatrix[1][0]) + " | " + String.format("%06d",confusionMatrix[1][1]) + " | " + String.format("%06d",confusionMatrix[1][2]) + " | " + String.format("%06d",confusionMatrix[1][3]));
			System.out.println("disagree : " + String.format("%06d",confusionMatrix[2][0]) + " | " + String.format("%06d",confusionMatrix[2][1]) + " | " + String.format("%06d",confusionMatrix[2][2]) + " | " + String.format("%06d",confusionMatrix[2][3]));
			System.out.println("discuss  : " + String.format("%06d",confusionMatrix[3][0]) + " | " + String.format("%06d",confusionMatrix[3][1]) + " | " + String.format("%06d",confusionMatrix[3][2]) + " | " + String.format("%06d",confusionMatrix[3][3]));
		} else {
			int[][] confusionMatrix = new int[4][4];
			float correct = 0.0F, incorrect = 0.0F;
			for(Headline headline:officialResults) {
				if(headline.correctlyClassed) {
					correct++;
				} else {
					incorrect++;
				}
				confusionMatrix[(int) headline.realClass][(int) headline.predictedClass]++;
			}
			System.out.println("Correct percentage was " + (float)correct/officialResults.size());
			System.out.println("Incorrect percentage was " + (float)incorrect/officialResults.size());
			System.out.println("\tunrelated | agree | disagree | discuss");
			System.out.println("unrelated: " + String.format("%06d",confusionMatrix[0][0]) + " | " + String.format("%06d",confusionMatrix[0][1]) + " | " + String.format("%06d",confusionMatrix[0][2]) + " | " + String.format("%06d",confusionMatrix[0][3]));
			System.out.println("agree    : " + String.format("%06d",confusionMatrix[1][0]) + " | " + String.format("%06d",confusionMatrix[1][1]) + " | " + String.format("%06d",confusionMatrix[1][2]) + " | " + String.format("%06d",confusionMatrix[1][3]));
			System.out.println("disagree : " + String.format("%06d",confusionMatrix[2][0]) + " | " + String.format("%06d",confusionMatrix[2][1]) + " | " + String.format("%06d",confusionMatrix[2][2]) + " | " + String.format("%06d",confusionMatrix[2][3]));
			System.out.println("discuss  : " + String.format("%06d",confusionMatrix[3][0]) + " | " + String.format("%06d",confusionMatrix[3][1]) + " | " + String.format("%06d",confusionMatrix[3][2]) + " | " + String.format("%06d",confusionMatrix[3][3]));
		}

	}
	
	/** 
	 * This generates a csv file, in accordance with the specifications laid 
	 * out by the scroing mechanism. The output is as follows, in a .csv file:
	 * "headline", "bodyID", "predicted stance", "classification score"
	 */
	public static void generateOutputCSV(String filename, int phase) throws IOException {
		
		if(phase == 1) {
			// CSVWriter writer = new CSVWriter(new FileWriter("test_results.csv"), '\t');
			CSVWriter writer = new CSVWriter(new FileWriter(filename), ',');
			// feed in your array (or convert your data to an array)
			System.out.println("Size of the testing results is " + testingResults.size());
			for(Headline headline : testingResults) {
				String[] entries = new String[3];
				entries[0]=headline.originalHeadline;
				entries[1]=headline.bodyID;
				entries[2]=headline.getPredictedStance();
				writer.writeNext(entries);
			}
			writer.close();
			
			
			BufferedWriter br = new BufferedWriter(new FileWriter("other" + filename));
			StringBuilder sb = new StringBuilder();
			  
			for(Headline headline:testingResults) {
				sb.append(headline.originalHeadline + '\t');
				sb.append(headline.bodyID + '\t');
				sb.append(headline.getPredictedStance() + '\n');
			}		
			  
			br.write(sb.toString());
			br.close();
		} else {
			// CSVWriter writer = new CSVWriter(new FileWriter("test_results.csv"), '\t');
			CSVWriter writer = new CSVWriter(new FileWriter(filename), ',');
			// feed in your array (or convert your data to an array)
			System.out.println("Size of the testing results is " + officialResults.size());
			for(Headline headline : officialResults) {
				String[] entries = new String[3];
				entries[0]=headline.originalHeadline;
				entries[1]=headline.bodyID;
				entries[2]=headline.getPredictedStance();
				writer.writeNext(entries);
			}
			writer.close();
			
			
			BufferedWriter br = new BufferedWriter(new FileWriter("other" + filename));
			StringBuilder sb = new StringBuilder();
			  
			for(Headline headline:officialResults) {
				sb.append(headline.originalHeadline + '\t');
				sb.append(headline.bodyID + '\t');
				sb.append(headline.getPredictedStance() + '\n');
			}		
			  
			br.write(sb.toString());
			br.close();
		}
	}

	/**************************************************************************
	 * The private inner class Document, which is used to store the lemmatized
	 * words of a news article.
	 **************************************************************************/
	static class MyDocument {

		// The lemmatized and tokenized Sentences, using CoreNLP
		List<Sentence> articleSentences = new ArrayList<Sentence>();
		// The article in one solid String
		String articleString = "";
		// The article as a list of Strings, one per sentence
		List<String> articleStrings = new ArrayList<String>();
		// The bodyID of the article, for reference into the HashMap
		String bodyID;

		// The document features that will be used to help determine if the
		// headline agrees, disagrees, or discusses the article
		float agreeSentences = 1.0F;
		float disagreeSentences = 1.0F;
		float discussSentences = 1.0F;

		// The positive/negative/neutral sentiment of the document, for use with
		// headline tagging
		float positive = 0.0F;
		float negative = 0.0F;
		float neutral = 0.0F;

		// The number of sentences that have a positive sentiment, a negative
		// sentiment, and the number of sentences total
		int positiveWords, negativeWords, totalWords;
		int positiveSentences, negativeSentences, neutralSentences;

		/*
		 *  Constructor for the Document class object
		 */
		public MyDocument(String theArticle, String theBodyID) {

			bodyID = theBodyID;
			Scanner articleScanner = new Scanner(theArticle);
			while (articleScanner.hasNextLine()) {
				String theSentence = articleScanner.nextLine();
				Sentence nlpSentence = new Sentence(theSentence);
				Sentence lemmaSentence = new Sentence(nlpSentence.lemmas());
				articleSentences.add(lemmaSentence);
				articleString += lemmaSentence.toString();
				// Sentiment Analysis
				Properties props = new Properties();
				props.setProperty("annotators", "tokenize, ssplit, pos, lemma, parse, sentiment");
				StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
				Annotation annotation = pipeline.process(theSentence);
				List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
				for (CoreMap sentence : sentences) {
					String sentiment = sentence.get(SentimentCoreAnnotations.SentimentClass.class);
					if (sentiment.equals("Negative")) {
						this.negativeSentences++;
					} else if (sentiment.equals("Positive")) {
						this.positiveSentences++;
					} else if (sentiment.equals("Neutral")) {
						this.neutralSentences++;
					}
				}
			}

			articleScanner.close();
			artcount++;
			Scanner stringScan = new Scanner(articleString);
			while (stringScan.hasNext()) {
				String word = stringScan.next();
				if (agreeWords.contains(word)) {
					this.addAgreeSentence();
				}
				if (disagreeWords.contains(word)) {
					this.addDisagreeSentence();
				}
				if (discussWords.contains(word)) {
					this.addDiscussSentence();
				}
				if (negWords.contains(word)) {
					this.negativeWords++;
				}
				if (posWords.contains(word)) {
					this.positiveWords++;
				}
				this.totalWords++;
				articleStrings.add(word);
			}

			stringScan.close();

			this.positive = this.positiveSentences / this.articleSentences.size();
			this.negative = this.negativeSentences / this.articleSentences.size();
			this.neutral = this.neutralSentences / this.articleSentences.size();

		}

		/*
		 *  The getter method that retrieves the sentiment of the body which is 
		 *  calculated based off the rates of each of the
		 *  different types of sentences encountered in the body
		 */
		public float getBodySentiment() {
			float returnVal = 0.0F;
			float positive = 1.0F, negative = 2.0F, neutral = 3.0F;
			if (this.positive > this.negative) {
				if (this.positive > this.neutral) {
					// The body is overwhelmingly positive
					returnVal = positive;
				}
			} else if (this.negative > this.positive) {
				if (this.negative > this.neutral) {
					// The body is overwhelmingly negative
					returnVal = negative;
				}
			} else if (this.neutral > this.positive) {
				if (this.neutral > this.negative) {
					// The body is overwhelmingly neutral
					returnVal = neutral;
				}
			}
			if (this.negative == this.positive) {
				// If equally negative and positive, it's neutral
				returnVal = neutral;
			}
			return returnVal;
		}

		/*
		 *  Used to get the body of the article as one large string primitive
		 */
		public String getArticleString() {
			return articleString;
		}

		/*
		 *  Used to get the sentences of the article as a List of string
		 *  primitives
		 */
		public List<String> getArticleStringList() {
			return articleStrings;
		}

		/* 
		 * Used to get the Body ID as a string primitive 
		 */
		public String getBodyIDString() {
			return this.bodyID;
		}

		/*
		 *  Incremented when a sentence contains a "discuss" word
		 */
		public void addDiscussSentence() {
			this.discussSentences++;
		}

		/*
		 *  Incremented when a sentence contains an "agree" word
		 */
		public void addAgreeSentence() {
			this.agreeSentences++;
		}

		/*
		 *  Incremented when a sentence contains a "disagree" word
		 */
		public void addDisagreeSentence() {
			this.disagreeSentences++;
		}

	}

	/**************************************************************************
	 * The private inner class Headline, which is used to store the lemmatized
	 * words of an article headline. Each Headline will also contain
	 * the score of each of the four class (unrelated, discusses, agrees, and
	 * disagrees) and will be assigned to the highest scoring of the four
	 * classes.
	 **************************************************************************/
	static class Headline {
		
		public int hashValue;
		boolean correctlyClassed = false;
		float realClass;
		double predictedClass;

		// The four individual scores that the document gets assigned
		// according to the headlines it is being checked against.
		boolean related;
		float agrees = 0.0F;
		float disagrees = 0.0F;
		float discusses = 0.0F;

		// The tf_idf score, used for determining relatedness
		// between a headline and a body of text
		float tf_idf;

		// The boolean positive score, will be used to determine whether or not
		// a headline is in general agreement with the sentiment of the article
		// body
		float positive = 0.0F;
		float negative = 0.0F;
		float neutral = 0.0F;
		float sentiment;

		// The bodyID features that this headline maps to
		float bodyAgreeSentences = 1.0F;
		float bodyDisagreeSentences = 1.0F;
		float bodyDiscussSentences = 1.0F;
		int bodyPositiveWords, bodyNegativeWords, bodyTotalWords;
		int positiveSentences, negativeSentences, neutralSentences;

		// String and Sentence values for the Headline Class structure
		String actualStance;
		String assignedStance;
		String bodyID;
		Sentence theHeadline;
		String headlineString = "";
		String originalHeadline;
		List<String> headlineStrings = new ArrayList<String>();

		/*
		 *  Constructor for the Headline Class object
		 */
		public Headline(String headline, String bodyID, String stance) {
			
			if(stance.equals("unrelated")) {
				this.realClass=0.0F;
			} else if(stance.equals("agree")) {
				this.realClass=1.0F;
			} else if(stance.equals("disagree")) {
				this.realClass=2.0F;
			} else {
				this.realClass=3.0F;
			}
			
			originalHeadline = headline;

			Scanner headlineScanner = new Scanner(headline);
			Sentence nlpHeadline = new Sentence(headline);
			theHeadline = new Sentence(nlpHeadline.lemmas());
			headlineString += theHeadline.toString();
			headlineScanner.close();
			this.bodyID = bodyID;
			actualStance = stance;
			// Sentiment Analysis
			Properties props = new Properties();
			props.setProperty("annotators", "tokenize, ssplit, pos, lemma, parse, sentiment");
			StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
			Annotation annotation = pipeline.process(headline);
			List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
			for (CoreMap sentence : sentences) {
				String sentiment = sentence.get(SentimentCoreAnnotations.SentimentClass.class);
				if (sentiment.equals("Negative")) {
					this.negativeSentences++;
				} else if (sentiment.equals("Positive")) {
					this.positiveSentences++;
				} else if (sentiment.equals("Neutral")) {
					this.neutralSentences++;
				}
			}

			switch (actualStance) {
			case "unrelated":
				unrelatedHeadlines++;
				break;
			case "discuss":
				discussesHeadlines++;
				break;
			case "agree":
				agreesHeadlines++;
				break;
			case "disagree":
				disagreesHeadlines++;
				break;
			default:
				break;
			}

			headcount++;
			Scanner stringScan = new Scanner(headlineString);

			while (stringScan.hasNext()) {
				String word = stringScan.next();
				headlineStrings.add(word);
			}
			stringScan.close();
		}

		/*
		 *  Relatedness -- related or unrelated, boolean value
		 */
		public void setRelatedness(boolean value) {
			this.related = value;
		}

		/*
		 *  TF-IDF score, for determining relatedness or unrelatedness
		 */
		public void setTFIDF(float new_tf_idf) {
			this.tf_idf = new_tf_idf;
		}

		/*
		 *  TF-IDF getter method
		 */
		public float getTFIDF() {
			return this.tf_idf;
		}

		/*
		 *  Returns the BodyID in a string format
		 */
		public String getBodyID() {
			return this.bodyID;
		}

		/*
		 *  Returns the Headline in a string format
		 */
		public String getHeadlineString() {
			return this.headlineString;
		}

		/*
		 *  Returns the Headline in a List of strings format
		 */
		public List<String> getHeadlineStrings() {
			return this.headlineStrings;
		}

		/*
		 *  Used to increment the "score" for how likely the class is "discusses"
		 */
		public void addDiscussPoints() {
			this.discusses += 15.0F;
		}

		/*
		 *  Used to increment the "score" for how likely the class is "disagree"
		 */
		public void addDisagreePoints() {
			this.disagrees += 1.0F;
		}

		/*
		 *  Used to increment the "score" for how likely the class is "agree"
		 */
		public void addAgreePoints() {
			this.agrees += 1.0F;
		}

		/*
		 *  Used to generate .arff entries for the Weka ML library
		 */
		public String generateArffEntry() {
			return this.tf_idf + "," + this.related;
		}
		
		public String getPredictedStance() {
			String returnVal = "";
			if(this.correctlyClassed) {
				returnVal=this.actualStance;
			} else {
				if(this.predictedClass==0.0F) {
					returnVal="unrelated"; 
				} else if(this.predictedClass==1.0F) {
					returnVal="agree";
				} else if(this.predictedClass==2.0F) {
					returnVal="disagree";
				} else if(this.predictedClass==3.0F) {
					returnVal="discuss";
				} else {
					returnVal="wrongClass";	
				}
			}
			return returnVal;
		}

		/*
		 *  Used to convert the boolean value of sentiment analysis into a float,
		 *  for Weka ML applications
		 */
		public float isPositive() {
			// TODO make this return a meaningful number
			return 0.0F;
		}
		
		public void setCorrectlyClassified() {
			this.correctlyClassed = true;
		}

		/*
		 * This method is used to determine the sentiment of this particular 
		 * headline. This value is used in conjunction with the sentiment of
		 * the body to create a "sentiment value" for the healdine
		 */
		public float getHeadlineSentiment() {
			float returnVal = 0.0F;
			float positive = 1.0F, negative = 2.0F, neutral = 3.0F;
			if (this.positive > this.negative) {
				if (this.positive > this.neutral) {
					// The body is overwhelmingly positive
					returnVal = positive;
				}
			} else if (this.negative > this.positive) {
				if (this.negative > this.neutral) {
					// The body is overwhelmingly negative
					returnVal = negative;
				}
			} else if (this.neutral > this.positive) {
				if (this.neutral > this.negative) {
					// The body is overwhelmingly neutral
					returnVal = neutral;
				}
			}
			if (this.negative == this.positive) {
				// If equally negative and positive, it's neutral
				returnVal = neutral;
			}
			return returnVal;
		}

	}

}
