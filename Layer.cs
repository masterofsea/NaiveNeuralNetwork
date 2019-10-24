using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NaiveNeuralNetwork
{
    public sealed class Layer
    {
        public Neuron[] Neurons { get; }
        public Int32 Count => Neurons?.Length ?? 0;

        public Layer(Neuron[] neurons, NeuronType type = NeuronType.Hidden)
        {
            Neurons = neurons;
        }
        public Double[] GetSignals()
        {
            var result = new Double[Neurons.Length];
            for(Int32 i = 0; i < Neurons.Length; ++i)
            {
                result[i] = Neurons[i].Output;
            }

            return result;
        }
    }
}
