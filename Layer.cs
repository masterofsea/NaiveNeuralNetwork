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
        public Int32 NeuronsCount => Neurons?.Length ?? 0;
        public NeuronType NType { get; set; }
        public Layer(Neuron[] neurons, NeuronType type = NeuronType.Hidden)
        {
            NType = type;
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
        public override string ToString()
        {
            return NType.ToString();
        }
    }
}
