using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NaiveNeuralNetwork
{
    class Topology
    {
        public Int32 InputCount { get; }
        public Int32 OutputCount { get; }
        public Int32[] HiddenLayers { get; }

        public Topology(Int32 inputCount, Int32 outputCount, params Int32[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            HiddenLayers = layers;
            
        }
    }
}
