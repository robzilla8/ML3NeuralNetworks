package mlProject3_NeuralNets;

// TODO: Auto-generated Javadoc
/**
 * The Class LogisticFunction.
 */
public class LogisticFunction implements ActivationFunction {

	/** The var. */
	private double var;
	
	/**
	 * Gets the output of the logistic function on the input
	 *
	 *	The result of applying this function gets stored in var
	 *
	 * @param input the input to be mapped to the logistic function
	 * @return the output of the input mapped to the logistic function
	 */
	@Override
	public double getOutput(double input) {
		// Change the function below, it's just a sigmoid for testing
		var = 1/(Math.pow(Math.E, -1*input)+1);
		return var;
	}

	/**
	 * Gets the derivative.
	 *
	 * @param input the input
	 * @return the derivative
	 */
	@Override
	public double getDerivative(double input) {
		// TODO Auto-generated method stub
		//System.out.printf("Getting derivative, returning %f%n", input*(1-input));
		return input*(1-input);
	}

	/**
	 * Gets the derivative of the logistic function at the same point as the last point that was fed
	 * through feed forward and had the logistic function applied to it.
	 *
	 * @return the derivative
	 */
	@Override
	public double getDerivative() {
		return getDerivative(var);
	}

}
