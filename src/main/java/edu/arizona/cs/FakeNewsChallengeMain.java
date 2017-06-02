package edu.arizona.cs;

//Standard Java imports for File I/O and utilities
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
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

	/*
	 * Booleans used for debugging purposes -- turn on individual boolean values
	 * to see print statements in their respective methods
	 */
	static boolean debug = false;
	static boolean articlePrint = false;
	static boolean headlinePrint = false;
	static boolean printTrain = false;
	static boolean printTest = false;
	static boolean priors = false;

	// HashMap of the body of articles
	static HashMap<String, MyDocument> articles = new HashMap<String, MyDocument>();
	// List of all headlines in the training set
	static List<Headline> headlines = new ArrayList<Headline>();
	// List of all the headlines in the testing set
	static List<Headline> testingHeadlines = new ArrayList<Headline>();

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
	static List<String> tripWords = Arrays.asList("-lrb-", "-rrb-", "NOT", "???", "!?", "--", "``");
	// List of words that are to be split because of lexical issues
	static List<String> splitWords = Arrays.asList("9/11-style", "15/month", "9/11", "Code/Red");
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
	// The classification model, specified as Naive Bayes
	static Classifier cModel;

	static int numberOfHeadlines;

	public static void main(String[] args) throws IOException, ParseException {

		// Strings for the relative paths to all three of the files
		String path_to_training_bodies, path_to_training_stances, path_to_testing_stances;
		path_to_training_bodies = getRelativePath(args[0]);
		path_to_training_stances = getRelativePath(args[1]);
		path_to_testing_stances = getRelativePath(args[2]);

		// Start a timer for the training
		Date startdate = new Date();
		float trainStartTime = startdate.getTime();
		System.out.println("Beginning training at " + startdate.toString());

		/*** Use the opencsv jar for reading the csv file ***/
		// Read the bodies of the testing data
		System.out.println("Preprocessing the " + args[0] + " file");
		CSVReader reader_bodies = new CSVReader(new FileReader(path_to_training_bodies));
		String[] nextLine_bodies;
		while ((nextLine_bodies = reader_bodies.readNext()) != null) {
			String bodyID = nextLine_bodies[0];
			String article = nextLine_bodies[1];
			createTheDocument(bodyID, article);
		}
		reader_bodies.close();

		// Read the headlines and stances of the testing data
		CSVReader reader_stances = new CSVReader(new FileReader(path_to_training_stances));
		String[] nextLine_stances;
		System.out.println("Preprocessing the " + args[1] + " file");
		while ((nextLine_stances = reader_stances.readNext()) != null) {
			String headline = nextLine_stances[0];
			String bodyID = nextLine_stances[1];
			String actualStance = nextLine_stances[2];
			createTheHeadline(headline, bodyID, actualStance);
		}
		reader_stances.close();

		DecimalFormat intFormat = new DecimalFormat("#");

		// Now that the training bodies and training headlines have been parsed
		// and lemmatized, begin Naive Bayes training for the set of training
		// data.
		StandardAnalyzer analyzer = new StandardAnalyzer();
		Directory index = new RAMDirectory();
		IndexWriterConfig config = new IndexWriterConfig(analyzer);
		IndexWriter w = new IndexWriter(index, config);
		// Add indexing for tf-idf similarity between Headline and Document
		for (String bodyID : articles.keySet()) {
			addDoc(w, bodyID, articles.get(bodyID).getArticleString());
		}

		numberOfHeadlines = headlines.size();

		preProcessForNBTraining();
		
		System.out.println("Beginning Naive Bayes training on preprocessed data");
		try {
			makeInstances();
		} catch (Exception e1) {
			e1.printStackTrace();
		}

		// End the timer for training, and calculate the time taken for training
		Date enddate = new Date();
		float trainEndTime = enddate.getTime();
		System.out.println("Ending training at " + enddate.toString());
		System.out.println("Time elapsed is " + (trainEndTime - trainStartTime) + " milliseconds.");
		// if (debug) {
		System.out.println("We had " + intFormat.format(artcount) + " articles");
		System.out.println("We had " + intFormat.format(headcount) + " headlines");
		// }

		System.out.println("Generating an .arff file for Weka");
		generateArffFile("test_stances_csc483583.arff");

		// Begin testing on the data set, keeping track of the important metrics
		// for performance
		Date testStart = new Date();
		float testStartTime = testStart.getTime();
		System.out.println("Beginning testing at " + testStart.toString());

		// Read the headlines and stances of the testing data
		CSVReader tester_stances = new CSVReader(new FileReader(path_to_testing_stances));
		String[] tester_nextLine_stances;
		System.out.println("Preprocessing the " + args[2] + " file");
		while ((tester_nextLine_stances = tester_stances.readNext()) != null) {
			String headline = tester_nextLine_stances[0];
			String bodyID = tester_nextLine_stances[1];
			String actualStance = tester_nextLine_stances[2];
			createTheTestHeadline(headline, bodyID, actualStance);
		}

		tester_stances.close();
		
		preProcessforNBTesting();
		
		System.out.println("Beginning Naive Bayes testing for preprocessed data");
		try {
			applyClassifyingModel();
		} catch (Exception e) {
			e.printStackTrace();
		}

		// Print out the time taken to conduct testing
		Date testEnd = new Date();
		float testEndTime = testEnd.getTime();
		System.out.println("Testing ended at " + testEnd.toString());
		System.out.println("Time elapsed is " + (testEndTime - testStartTime) + " milliseconds");
		
		generateOutputCSV();
		getFinalScore();

	}

	/**
	 * The creation and instantiation of the Instances that are added to the
	 * training data set before a classifier is derived from the values that are
	 * being used
	 **/
	public static void makeInstances() throws Exception {

		// Declare the first Attribute, the tf_idf
//		Attribute Attribute1 = new Attribute("tf-idf");
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
		// fvClassVal.add("unrelated");
		fvClassVal.add("agree");
		fvClassVal.add("disagree");
		fvClassVal.add("discuss");
		Attribute ClassAttribute = new Attribute("the_class", fvClassVal);

		// Declare the feature vector
		ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>();
//		fvWekaAttributes.add(Attribute1);
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

		for (Headline headline : headlines) {
			if (!headline.getBodyID().equals("Body ID")) {
				if (headline.related) {
					// Create the instance
					Instance iExample = new DenseInstance(6);
//					iExample.setValue((Attribute) fvWekaAttributes.get(0), headline.getTFIDF());
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
		System.out.println(cModel);

	}

	/**
	 * The application of the Weka Naive Bayes Machine Learning algorithm, using
	 * several distinct attributes for supervised learning
	 **/
	public static void applyClassifyingModel() throws Exception {

		// Declare the first Attribute, the tf_idf
//		Attribute Attribute1 = new Attribute("tf-idf");
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
		// fvClassVal.add("unrelated");
		fvClassVal.add("agree");
		fvClassVal.add("disagree");
		fvClassVal.add("discuss");
		Attribute ClassAttribute = new Attribute("the_class", fvClassVal);

		// Declare the feature vector
		ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>();
//		fvWekaAttributes.add(Attribute1);
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

		HashMap<Integer, Headline> mistakeMap = new HashMap<Integer, Headline>();
		ArrayList<Integer> hashcodes = new ArrayList<Integer>();
		
		int counter = 0;
		for (Headline headline : testingHeadlines) {
			if (!headline.getBodyID().equals("Body ID")) {
				if (headline.related) {
					System.out.println(counter + " : " + headline.headlineString + " | " + headline.actualStance);
					System.out.println();
					// Keeping track of any mistakes made by the classifier
					headline.hashValue = headline.hashCode();
					hashcodes.add(headline.hashValue);
					mistakeMap.put(headline.hashValue, headline);
					// Create the instance
					Instance iExample = new DenseInstance(6);
//					iExample.setValue((Attribute) fvWekaAttributes.get(0), headline.getTFIDF());
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
			counter++;
		}
		// Test the model
		Evaluation eTest = new Evaluation(testing_set);
		eTest.evaluateModel(cModel, testing_set);
		
		
		// TODO Work on developing a viable output in proper format
		
		 ArrayList<Prediction> Predictions = eTest.predictions();
		 int predictionCounter = 0;
//		 int predictionSize = Predictions.size();
//		 System.out.println("Number of predicitons is " + Predictions.size());
		 for (Prediction predict : Predictions) {
//			 System.out.println(predict.toString());
			 if (predict.predicted() == predict.actual()) {
				 int index = hashcodes.get(predictionCounter);
				 Headline correctHeadline = mistakeMap.get(index);
				 System.out.println("Correctly classified:");
				 System.out.println(correctHeadline.headlineString);
				 System.out.println(predict.toString());
				 System.out.println(predict.predicted() + " and was " + predict.actual()); //TODO Look here!
				 System.out.println();
				 correctHeadline.correctlyClassed=true;
			 } else {
				 int index = hashcodes.get(predictionCounter);
				 Headline correctHeadline = mistakeMap.get(index);
				 System.out.println("Correctly classified:");
				 System.out.println(correctHeadline.headlineString);
				 System.out.println(predict.toString());
				 System.out.println(predict.predicted() + " and was " + predict.actual()); //TODO Look here!
				 System.out.println();
			 }
			 predictionCounter++;
		 }

		String classDeets = eTest.toClassDetailsString();
		System.out.println("Class Details " + classDeets);

		// Print the result à la Weka explorer:
		String strSummary = eTest.toSummaryString();
		System.out.println("Performance Summary " + strSummary);

		// Get the confusion matrix
//		double[][] cmMatrix = eTest.confusionMatrix();
		

		// Specify that the instance belong to the training set
		// in order to inherit from the set description
		// iUse.setDataset(isTrainingSet);

		// Get the likelihood of each classes
		// fDistribution[0] is the probability of being “positive”
		// fDistribution[1] is the probability of being “negative”
		// double[] fDistribution = cModel.distributionForInstance(iUse);

	}

	/**
	 * Used to generate a .arff file, which is a unique file extension used
	 * extensively by the Weka ML Library
	 **/
	public static void generateArffFile(String filename) {

		BufferedWriter bw = null;
		FileWriter fw = null;
		int loadScreen = Math.round(numberOfHeadlines / 75);
		int counter = 0;

		try {
			String header = "@RELATION First\n\n@ATTRIBUTE tf_idf NUMERIC\n@ATTRIBUTE related {true,false}\n\n@data\n";

			fw = new FileWriter(filename);
			bw = new BufferedWriter(fw);
			bw.write(header);
			for (Headline headline : headlines) {
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
	public static void preProcessforNBTesting() throws IOException, ParseException {
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
						if (debug && printTrain) {
							System.out.println(headline.actualStance);
							System.out.println(headline.getHeadlineString());
							System.out.println("tf_idf score is " + topHits[0].score + "\n");
						}
					} else {
						// QueryParser tf-idf used for ML algorithms
						headline.setTFIDF(tf_idf);
						headline.setRelatedness(false);
						if(headline.actualStance.equals("unrelated")) {
							headline.correctlyClassed=true;
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
	 * Used to determine the values of the selected features which will be used
	 * in the classifying model
	 **/
	public static void preProcessForNBTraining() throws IOException, ParseException {

		if (debug) {
			// Debug print statement
			System.out.println("In Naive Bayes training");
		}

		// Preprocess for each headline in the list
		for (Headline headline : headlines) {
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
				if (debug && printTrain) {
					System.out.println(headline.actualStance);
					System.out.println(headline.getHeadlineString());
					System.out.println("tf_idf score is " + topHits[0].score + "\n");
				}
			} else {
				// QueryParser tf-idf used for ML algorithms
				headline.setTFIDF(tf_idf);
				headline.setRelatedness(false);
				if(headline.actualStance.equals("unrelated")) {
					headline.correctlyClassed=true;
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
	public static void createTheDocument(String bodyID, String theArticle) {
		theArticle = theArticle.replaceAll("\\r\\n|\\r|\\n", " ");
		MyDocument d = null;
		if (articles.containsKey(bodyID)) {
			d = (MyDocument) articles.get(bodyID);
		} else {
			d = new MyDocument(theArticle, bodyID);
			articles.put(bodyID, d);
		}
	}

	/**
	 * Called to create a headline for the training headlines List of headlines
	 **/
	public static void createTheHeadline(String headline, String bodyID, String actualStance) {
		Headline h = new Headline(headline, bodyID, actualStance);
		headlines.add(h);
	}

	/**
	 * Called to create a headline for the testingHeadlines List of headlines
	 **/
	public static void createTheTestHeadline(String headline, String bodyID, String actualStance) {
		Headline h = new Headline(headline, bodyID, actualStance);
		testingHeadlines.add(h);
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
	public static void getFinalScore() {
		
		int[][] confusionMatrix;
		//TODO

	}
	
	/** 
	 * This generates a csv file, in accordance with the specifications laid 
	 * out by the scroing mechanism. The output is as follows, in a .csv file:
	 * "headline", "bodyID", "predicted stance", "classification score"
	 */
	public static void generateOutputCSV() throws IOException {
		//TODO
		
		CSVWriter writer = new CSVWriter(new FileWriter("yourfile.csv"), '\t');
		// feed in your array (or convert your data to an array)
		for(Headline headline:testingHeadlines) {
			String[] entries = (headline.headlineString+"#"+headline.bodyID+"#"+headline.getPredictedStance()).split("#");
			System.out.println(entries[0]+ " : " + entries[1] + " : " + entries[2]);
			writer.writeNext(entries);
		}
		writer.close();
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
		float predictedClass;

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
		List<String> headlineStrings = new ArrayList<String>();

		/*
		 *  Constructor for the Headline Class object
		 */
		public Headline(String headline, String bodyID, String stance) {
			
			//TODO make the realClass variable here!!

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

			if (debug && headlinePrint) {
				System.out.println("Headline of " + theHeadline + "\n");
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
			//TODO make the return of this a String of the predicted type
			// i.e. turn a numeric value into a string
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
