using System;

namespace NeuralNetworks_1
{
    class nn3S
    {
        double[,] wih;
        double[,] who;

        int inodes;
        int hnodes;
        int onodes;

        double learningRate;


        double[] hidden_inputs =
            Array.Empty<double>();

        double[] hidden_outputs =
            Array.Empty<double>();

        double[] final_inputs =
            Array.Empty<double>();

        double[] final_outputs =
            Array.Empty<double>();

        double[] output_errors =
            Array.Empty<double>();

        double[] hidden_errors =
            Array.Empty<double>();


        public double[,] WIH
        {
            get { return wih; }
        }

        public double[,] WHO
        {
            get { return who; }
        }

        public double[] Hidden_inputs
        {
            get { return hidden_inputs; }
        }

        public double[] Hidden_outputs
        {
            get { return hidden_outputs; }
        }

        public double[] Final_inputs
        {
            get { return final_inputs; }
        }

        public double[] Final_outputs
        {
            get { return final_outputs; }
        }

        public double[] Output_errors
        {
            get { return output_errors; }
        }

        public double[] Hidden_errors
        {
            get { return hidden_errors; }
        }


        public nn3S(
            int inodes,
            int hnodes,
            int onodes,
            double learningRate)
        {
            this.inodes =
                inodes;

            this.hnodes =
                hnodes;

            this.onodes =
                onodes;

            this.learningRate =
                learningRate;


            wih =
                new double[
                    inodes,
                    hnodes
                ];

            who =
                new double[
                    hnodes,
                    onodes
                ];


            createWeightMatrizes();
        }


        private void createWeightMatrizes()
        {
            wih =
                new double[
                    inodes,
                    hnodes
                ];

            who =
                new double[
                    hnodes,
                    onodes
                ];


            // Input -> Hidden
            for (int j = 0;
                 j < hnodes;
                 j++)
            {
                for (int i = 0;
                     i < inodes;
                     i++)
                {
                    Random weight_ih =
                        new Random(
                            Guid.NewGuid()
                                .GetHashCode()
                        );

                    wih[i, j] =
                        (
                            weight_ih
                                .NextDouble()
                            - 0.5
                        )
                        * 2.0;
                }
            }


            // Hidden -> Output
            for (int j = 0;
                 j < onodes;
                 j++)
            {
                for (int i = 0;
                     i < hnodes;
                     i++)
                {
                    Random weight_ho =
                        new Random(
                            Guid.NewGuid()
                                .GetHashCode()
                        );

                    who[i, j] =
                        (
                            weight_ho
                                .NextDouble()
                            - 0.5
                        )
                        * 2.0;
                }
            }
        }


        public void queryNN(
            double[] inputs)
        {
            nnMath nnMathO =
                new nnMath();


            hidden_inputs =
                nnMathO.matrixMult(
                    wih,
                    inputs
                );


            hidden_outputs =
                nnMathO.activationFunction(
                    hidden_inputs
                );


            final_inputs =
                nnMathO.matrixMult(
                    who,
                    hidden_outputs
                );


            final_outputs =
                nnMathO.activationFunction(
                    final_inputs
                );
        }


        public void train(
            double[] inputs,
            double[] targets)
        {
            nnMath nnMathO =
                new nnMath();


            // Forward Pass
            queryNN(inputs);


            // Output Error
            output_errors =
                new double[
                    onodes
                ];

            for (int i = 0;
                 i < onodes;
                 i++)
            {
                output_errors[i] =
                    targets[i]
                    - final_outputs[i];
            }


            // Hidden Error
            double[,] whoTransposed =
                nnMathO.transpose(
                    who
                );


            hidden_errors =
                nnMathO.matrixMult(
                    whoTransposed,
                    output_errors
                );


            // WHO aktualisieren
            double[] outputGradient =
                new double[
                    onodes
                ];

            for (int i = 0;
                 i < onodes;
                 i++)
            {
                outputGradient[i] =
                    output_errors[i]
                    * final_outputs[i]
                    * (
                        1.0
                        - final_outputs[i]
                    );
            }


            double[,] deltaWho =
                nnMathO.fullMatrixMult(
                    hidden_outputs,
                    outputGradient
                );


            deltaWho =
                nnMathO.matrixScale(
                    deltaWho,
                    learningRate
                );


            who =
                nnMathO.matrixSum(
                    who,
                    deltaWho
                );


            // WIH aktualisieren
            double[] hiddenGradient =
                new double[
                    hnodes
                ];


            for (int i = 0;
                 i < hnodes;
                 i++)
            {
                hiddenGradient[i] =
                    hidden_errors[i]
                    * hidden_outputs[i]
                    * (
                        1.0
                        - hidden_outputs[i]
                    );
            }


            double[,] deltaWih =
                nnMathO.fullMatrixMult(
                    inputs,
                    hiddenGradient
                );


            deltaWih =
                nnMathO.matrixScale(
                    deltaWih,
                    learningRate
                );


            wih =
                nnMathO.matrixSum(
                    wih,
                    deltaWih
                );
        }


        public void setWIHMatrix(
            double[,] matrix)
        {
            wih = matrix;
        }


        public void setWHOMatrix(
            double[,] matrix)
        {
            who = matrix;
        }
    }
}