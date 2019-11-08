package mlProject3_NeuralNets;

// TODO: Auto-generated Javadoc
/**
 * The Interface ActivationFunction.
 */
public interface ActivationFunction {
	
	/**
	 * Gets the output of applying some activation function to the input
	 *
	 * @param input the input
	 * @return the output
	 */
	public double getOutput(double input);
	
	/**
	 * Gets the derivative of the activation function at the point input
	 *
	 * @param input the input
	 * @return the derivative
	 */
	public double getDerivative(double input);
	
	/**
	 * Gets the derivative at the point last fed through the network via feedforward
	 *
	 * @return the derivative
	 */
	public double getDerivative();

}
