using System;

namespace NeuralNetworks_1;

class nnMath
{
    public double[] matrixMult(
        double[,] matrix,
        double[] vector)
    {
        int inputs =
            matrix.GetLength(0);

        int outputs =
            matrix.GetLength(1);

        double[] result =
            new double[outputs];

        for (int j = 0;
             j < outputs;
             j++)
        {
            double sum = 0;

            for (int i = 0;
                 i < inputs;
                 i++)
            {
                sum +=
                    matrix[i, j]
                    * vector[i];
            }

            result[j] = sum;
        }

        return result;
    }


    public double[,] fullMatrixMult(
        double[] vec1,
        double[] vec2)
    {
        double[,] result =
            new double[
                vec1.Length,
                vec2.Length
            ];

        for (int i = 0;
             i < vec1.Length;
             i++)
        {
            for (int j = 0;
                 j < vec2.Length;
                 j++)
            {
                result[i, j] =
                    vec1[i]
                    * vec2[j];
            }
        }

        return result;
    }


    public double[,] matrixSum(
        double[,] mat1,
        double[,] mat2)
    {
        int rows =
            mat1.GetLength(0);

        int columns =
            mat1.GetLength(1);

        double[,] result =
            new double[
                rows,
                columns
            ];

        for (int i = 0;
             i < rows;
             i++)
        {
            for (int j = 0;
                 j < columns;
                 j++)
            {
                result[i, j] =
                    mat1[i, j]
                    + mat2[i, j];
            }
        }

        return result;
    }


    public double[] vectorMult(
        double[] vec1,
        double[] vec2)
    {
        double[] result =
            new double[
                vec1.Length
            ];

        for (int i = 0;
             i < vec1.Length;
             i++)
        {
            result[i] =
                vec1[i]
                * vec2[i];
        }

        return result;
    }


    public double[,] transpose(
        double[,] inputMat)
    {
        int rows =
            inputMat.GetLength(0);

        int columns =
            inputMat.GetLength(1);

        double[,] result =
            new double[
                columns,
                rows
            ];

        for (int i = 0;
             i < rows;
             i++)
        {
            for (int j = 0;
                 j < columns;
                 j++)
            {
                result[j, i] =
                    inputMat[i, j];
            }
        }

        return result;
    }


    public double[,] matrixScale(
        double[,] inputMat,
        double scale)
    {
        int rows =
            inputMat.GetLength(0);

        int columns =
            inputMat.GetLength(1);

        double[,] result =
            new double[
                rows,
                columns
            ];

        for (int i = 0;
             i < rows;
             i++)
        {
            for (int j = 0;
                 j < columns;
                 j++)
            {
                result[i, j] =
                    inputMat[i, j]
                    * scale;
            }
        }

        return result;
    }


    public double[] activationFunction(
        double[] inputs)
    {
        double[] result =
            new double[
                inputs.Length
            ];

        for (int i = 0;
             i < inputs.Length;
             i++)
        {
            result[i] =
                1.0 /
                (
                    1.0
                    + Math.Exp(
                        -inputs[i]
                    )
                );
        }

        return result;
    }
}