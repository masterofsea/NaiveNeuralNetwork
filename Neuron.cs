using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NaiveNeuralNetwork
{
    public sealed class Neuron
    {
        public Double[] Weights { get; }
        public Double[] Inputs { get; }
        public NeuronType NType { get; }
        public double Output { get; private set; }
        public Double Delta { get; private set; }
        public Neuron(Int32 inputCount, NeuronType type= NeuronType.Hidden)
        {
            Random rnd = new Random();
            NType = type;
            Weights = new Double[inputCount];
            if (NType != NeuronType.Input)
            {
                for (Int32 i = 0; i < inputCount; ++i)
                    Weights[i] = rnd.NextDouble() * 2 - 1;                
            }
            else
            {
                for (Int32 i = 0; i < inputCount; ++i)
                    Weights[i] = 1;
            }
            
            Inputs = new Double[inputCount];

        }       
        public Double FeedForward(Double[] inputs)
        {
            if (inputs.Length != Weights.Length) throw new Exception("Массив входных значений должен быть равен числу входов");
            for(Int32 i = 0; i < Inputs.Length; ++i)
            {
                Inputs[i] = inputs[i];
            }
            Double accumulator = 0.0;
            for (Int32 i = 0; i < inputs.Length; ++i) accumulator += inputs[i] * Weights[i];
            if(NType != NeuronType.Input)
            {
                Output = Sigm(accumulator);
            }
            else
            {
                Output = accumulator;
            }
            
            return Output;
        }
        public void Learn(Double error, Double learningRate)
        {
            if (NType == NeuronType.Input) return;

            Delta = error * DerivativeOfSigm(Output);
            for(Int32 i = 0; i < Weights.Length; ++i)
            {
                Weights[i] -= Inputs[i] * Delta * learningRate;
            }
       }
        private Double Sigm(Double value)
        {
            return (1.0 / (1 + Math.Pow(Math.E, -value)));
        }
        private Double DerivativeOfSigm(Double value)
        {
            var sigmoid = Sigm(value);
            return sigmoid * (1 - sigmoid);
        }


        public override string ToString()
        {
            return Output.ToString();
        }

        

    }
}
