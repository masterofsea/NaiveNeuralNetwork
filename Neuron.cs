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
        public NeuronType NType { get; }
        public double Output { get; private set; }        
        public Neuron(Int32 inputCount, NeuronType type= NeuronType.Hidden)
        {
            Random rnd = new Random();
            NType = type;
            Weights = new Double[inputCount];
            
        }       
        public Double FeedForward(Double[] inputs)
        {
            if (inputs.Length != Weights.Length) throw new Exception("Массив входных значений должен быть равен числу входов");
            Double accumulator = 0.0;
            for (Int32 i = 0; i < inputs.Length; ++i) accumulator += inputs[i] * Weights[i];
            Output = Sigm(accumulator);
            return Output;
        }

        private Double Sigm(Double value)
        {
            return (1.0 / (1 + Math.Pow(Math.E, -value)));
        }
        public override string ToString()
        {
            return Output.ToString();
        }

    }
}
