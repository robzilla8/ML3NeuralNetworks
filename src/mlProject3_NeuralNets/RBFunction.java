import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.io.*;

public class RBFunction extends Node{

    EuclidianDistance eu = new EuclidianDistance();

    protected static ArrayList<ArrayList<Object>> hiddenLayer = new ArrayList<ArrayList<Object>>();

    protected ArrayList<Object> product = new ArrayList<Object>();

    // read the nodes from the CSV file
    public static void RBFunctionn(String filename) {

        try {

            Scanner fileInput = new Scanner(new File("file.txt"));

            // loop when there's a line to read
                while (fileInput.hasNext()){
                    // store this line as a string
                    String line = fileInput.nextLine();
                    ArrayList<Object> prototype = new ArrayList<Object>();

                    //Stop reading once the word End is encountered
                    if (line.equals("End")){
                        break;
                    }


                    else {

                            //System.out.println(line);

                        // split the line
                            String[] input = line.split(",");

                            // store the vector in an arraylist
                            for (int p = 0; p < input.length; p++) {
                                Object element = input[p];
                                prototype.add(element);
                            }
                            //System.out.println("prototype is" + Arrays.toString(prototype.toArray()));

                            // add the vector to another arraylist
                            hiddenLayer.add(prototype);
                            //System.out.println(Arrays.toString(hiddenLayer.toArray()));
                        }



            }

                fileInput.close();

    } catch(IOException e){
            System.out.println("Error");
        }
    }

    // get the distance between the testnode and each node in the hidden layer
    public void getDis(ArrayList<Object> test) {

        double dis = 0.0;
        Random rand = new Random();


        for (int i = 0; i < hiddenLayer.size(); i++) {

            // weights should be tuned with backpropagation
            double weight = rand.nextDouble()*0.02 - 0.01;
            dis = eu.getDistance(hiddenLayer.get(i), test);

            // multiply the distance by the tuned weight -presumably
            double out = weight * dis;
            Object p = (Object) out;

            // add the product to an array
            product.add(p);
        }
    }

}
