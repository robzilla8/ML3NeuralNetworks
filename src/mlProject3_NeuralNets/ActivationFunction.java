package mlProject3_NeuralNets;

public interface ActivationFunction {
	
	public double getOutput(double input);
	
	public double getDerivative(double input);
	
	public double getDerivative();

}
