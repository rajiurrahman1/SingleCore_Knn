
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author rajiurrahman1
 */
public class Knn {
    
    public float[][] readData(String fileNamePath, int numRow, int numCol) throws FileNotFoundException, IOException{
        
        File file = new File(fileNamePath);
        FileReader fileReader = new FileReader(file);
        BufferedReader buffReader = new BufferedReader(fileReader);
        
        float[][] dataMatrix = new float[numRow][numCol+1];
        String line; 
        String[] arr;
        int count1 =0;
        while ((line = buffReader.readLine()) != null) {                                
            arr = line.split(" ");
            float f1 = (float)0.0;
            for(int j=0; j<=numCol; j++){
                
                f1 = Float.parseFloat(arr[j].trim());
                dataMatrix[count1][j] = f1;
            }            
            count1 ++;
        }
        return dataMatrix;
    }
    
    public void dumpMatrix(float[][] dataMatrix){
        for(int i=0; i<dataMatrix.length; i++){
            for(int j=0; j<dataMatrix[0].length; j++){
                System.out.print(dataMatrix[i][j]+"  ");
            }
            System.out.print("\n");
        }
    }
    
    public void dumpMatrixInt(int[][] dataMatrix){
        for(int i=0; i<dataMatrix.length; i++){
            for(int j=0; j<dataMatrix[0].length; j++){
                System.out.print(dataMatrix[i][j]+"  ");
            }
            System.out.print("\n");
        }
    }
    
    private float computeRowDistance(int i, int j, int numCol, float[][] ref, float[][] test){
        float distance = (float) 0.0;
        for (int k=0; k<numCol; k++){
            distance += Math.pow(  (test[i][k] - ref[j][k]), 2 );
        }
        distance = distance/(float)numCol;
        distance = (float) Math.sqrt(distance);
        return distance;
    }
    
    private float [][] createSimilarityMatrix(int numRowRef, int numRowTest, int numCol, float[][] ref, float[][] test){
        Knn k1 = new Knn();
        float[][] similarityMatrix = new float[numRowTest][numRowRef];
        for (int i=0; i<numRowTest; i++){
            for (int j=0; j<numRowRef; j++){
                similarityMatrix[i][j] = k1.computeRowDistance(i, j, numCol, ref, test);
            }
//            System.out.println("Similarity Matrix - test row: "+i);
        }
        return similarityMatrix;
    }
    
    public int getMinIndex(float[][] similarityMatrix, int i, int numRowRef){
        //find the index of the minimum value of similarityMatrix[i][]
        int minIndex = 0;
        float minDistance = (float)1;
        
        for (int j=0; j<numRowRef; j++){
            if(similarityMatrix[i][j] < minDistance  ){
                minDistance = similarityMatrix[i][j];
                minIndex = j;
            }
        }
        return minIndex;
    }
    
    private int[][] calculateNearestNeighbor(int numRowTest, int numRowRef, int k, float[][] similarityMatrix){
        int[] testPredictedClassLabel = new int[numRowTest];
        int[][] nearestNeighborIndices = new int[numRowTest][k];
        
        for(int i=0; i<numRowTest; i++){
            for (int j=0; j<k; j++){
                int minIndex = getMinIndex(similarityMatrix, i, numRowRef);
                nearestNeighborIndices[i][j] = minIndex;
                similarityMatrix[i][minIndex] = (float)1;
            }
        }        
        return nearestNeighborIndices;
    }
    
    private int[] predictClassLabels(int[][] nearestNeighbors, float[][] trainData, int numRowTest, int numCol, int k){
        int[] testPredictedClassLabel = new int[numRowTest];
        int predictedLabel = 0;
        
        for (int i=0; i<numRowTest; i++){
            int sumLabel = (int)0;
            for (int j=0; j<k; j++){
                int index = nearestNeighbors[i][j];
                sumLabel += trainData[index][numCol];
            }
            if(sumLabel > k/2){
                predictedLabel = 1;
                testPredictedClassLabel[i] = predictedLabel;
            }
            else{
                predictedLabel = 0;
                testPredictedClassLabel[i] = predictedLabel;
            }
        }
        return testPredictedClassLabel;
    }
    
    private float calculatePredictionAccuracy(int[] testPredictedClassLabel, float[][] testData, int numRowTest, int numCol){
        int accurateCount = 0;
        float accuracy = (float)0;
        for(int i=0; i<numRowTest; i++){
            if(testPredictedClassLabel[i] == testData[i][numCol]){
                accurateCount += 1;
            }
        }
        accuracy = 100*accurateCount/(float)numRowTest;
        
        return accuracy;
    }
    
    public static void main(String[] args) throws IOException{
        System.out.println("Hello world! ");
        long startTime = System.currentTimeMillis();
        Knn objKnn = new Knn();
        
        //catch arguments from command line
        String trainDataFileName = (String)args[0].trim();
        String testDataFileName = (String)args[1].trim();
        int numRowTrain = Integer.parseInt(args[2]);
        int numRowTest = Integer.parseInt(args[3]);
        int numCol = Integer.parseInt(args[4]);
        int k = Integer.parseInt(args[5]);

        
        
        float[][] trainData = objKnn.readData(trainDataFileName, numRowTrain, numCol);
        System.out.println("*********\nTraining Data read completed\n");
        float[][] testData = objKnn.readData(testDataFileName, numRowTest, numCol);
        System.out.println("Test Data read completed\n");
        
        float[][] similarityMatrix = objKnn.createSimilarityMatrix(numRowTrain, numRowTest, numCol, trainData, testData);
        System.out.println("\n*********\nSimilarity Matrix calculation done");
        int [][] nearestNeighbors = objKnn.calculateNearestNeighbor(numRowTest, numRowTrain, k, similarityMatrix);
        System.out.println("\nNearest Neighbor calculation done");
        
        int[] testPredictedClassLabel = objKnn.predictClassLabels(nearestNeighbors, trainData, numRowTest, numCol, k);
        System.out.println("\nClass label prediction done");
        //objKnn.dumpMatrixInt(nearestNeighbors);
        
        float predictionAccuracy = objKnn.calculatePredictionAccuracy(testPredictedClassLabel, testData, numRowTest, numCol);
        
        System.out.println( "\n\n*********\nPrediction Accuracy: " + predictionAccuracy );
        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Time taken: "+ totalTime+ " miliseconds\n");
    }
    
}
