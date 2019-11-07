package mlProject3_NeuralNets;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNetTestDriver {
	
	public static void main(String[] args) {
		test(1);
	}
	
	private static void test(int iterations) {
		for (int j = 0; j < iterations; j++) {
			Random r = new Random();
			ArrayList<ArrayList<Double>> testData = new ArrayList<ArrayList<Double>>();
			for (int i = 0; i < 35; i++) {
				double randInt1 = (r.nextInt(200) + 200) /*/ 100.0*/; // random value between 0.25 and 1
				double randInt2 = r.nextInt(200) - 300 /*/ 100.0*/; // random value between 0.0 and 0.5
				double randInt3 = randInt1 - randInt2;
			//	randInt3 = 0.025;
				ArrayList<Double> arr = new ArrayList<Double>();
				arr.add(randInt1);
				arr.add(randInt2);
				arr.add(randInt3);
				testData.add(arr);
			}
			
			// Trying xor
//			ArrayList<Double> test1 = new ArrayList<Double>();
//			ArrayList<Double> test2 = new ArrayList<Double>();
//			ArrayList<Double> test3 = new ArrayList<Double>();
//			ArrayList<Double> test4 = new ArrayList<Double>();
//			test1.add(1.0);
//			test1.add(1.0);
//			test1.add(0.0);
//			
//			test2.add(1.0);
//			test2.add(0.0);
//			test2.add(1.0);
//			
//			test3.add(0.0);
//			test3.add(1.0);
//			test3.add(1.0);
//			
//			test4.add(0.0);
//			test4.add(0.0);
//			test4.add(0.0);
//			
//			testData.add(test1);
//			testData.add(test2);
//			testData.add(test3);
//			testData.add(test4);
			
			NeuralNetwork nn = new NeuralNetwork(2, 4, 2, 1, false, 0.04);
			nn.printAllNodeInfo();
			ArrayList<Double> dubs = new ArrayList<Double>();
			for (int i = 1; i <= 1000; i++) {
				dubs.add(r.nextDouble());
			}
			for (int i = 0; i < 3; i++) {
				System.out.printf("BackProp iteration %d%n", i);
				ArrayList<Integer> indeces = new ArrayList<Integer>();
				for (int p = 0; p < testData.size(); p++) {
					indeces.add(p);
				}
				for (int p = 0; p < testData.size(); p++) {
				//ArrayList<Double> curVector = testData.get(r.nextInt(testData.size()));
					int randIndex = r.nextInt(indeces.size());
					int index = indeces.get(randIndex);
					indeces.remove(randIndex);
					ArrayList<Double> curVector = testData.get(index);
					nn.feedForward(curVector);
					ArrayList<Double> target = new ArrayList<Double>();
					target.add(curVector.get(2));
					nn.backProp(target);
					System.out.printf("%n%n%n%n%n");
				}
			}
			
			double pureError = 0;
			for (ArrayList<Double> vector : testData) {
				ArrayList<Double> output = nn.feedForward(vector);
				pureError += Math.abs(output.get(0) - vector.get(2));
				System.out.printf("	Target: %f%n", vector.get(2));
	//			nn.printAllNodeInfo();
			}
			//nn.printAllNodeInfo();
			System.out.printf("Pure error = %f%n", pureError);
			
			double avg = 0;
			for (int i = 0; i < testData.size(); i++) {
				avg += testData.get(i).get(testData.get(i).size() - 1);
			}
			avg = avg / testData.size();
			System.out.printf("Average: %f%n", avg);
		}
	}
}