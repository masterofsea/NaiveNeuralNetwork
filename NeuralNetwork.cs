using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NaiveNeuralNetwork
{
    class NeuralNetwork
    {
        public List<Layer> Layers { get; }
        public Topology Topology{ get; }

        

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayers();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new Neuron[Topology.OutputCount];
            var lastLayerNeuronsCount = Layers.Last().NeuronsCount;
            for (Int32 i = 0; i < outputNeurons.Length; ++i)
            {
                outputNeurons[i] = new Neuron(lastLayerNeuronsCount, NeuronType.Input);
            }

            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }
        private void CreateHiddenLayers()
        {
            for(Int32 i = 0; i < Topology.HiddenLayers.Length; ++i)
            {
                var hiddenNeurons = new Neuron[Topology.HiddenLayers[i]];
                var lastLayerNeuronsCount = Layers.Last().NeuronsCount;

                for (Int32 j = 0; j < hiddenNeurons.Length; ++j)
                {
                    hiddenNeurons[i] = new Neuron(lastLayerNeuronsCount, NeuronType.Input);
                }

                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }            
        }
        private void CreateInputLayers()
        {
            var inputNeurons = new Neuron[Topology.InputCount];
            for(Int32 i = 0; i < inputNeurons.Length; ++i)
            {
                inputNeurons[i] = new Neuron(1, NeuronType.Input);
            }

            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }
        public Neuron FeedForward(Double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();
            if (Topology.OutputCount == 1) return Layers.Last().Neurons[0];
            else return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
        }
        public Double Learn(List<Tuple<Double, Double[]>> dataset, Int32 epoch)
        {
            var error = 0.0;
            for(Int32 i = 0; i < epoch; ++i)
            {
                foreach(var data in dataset)
                {
                    error += BackPropagation(data.Item1, data.Item2);
                }
            }

            var result = error / epoch;
            return result;
        }
        private Double BackPropagation(Double expected, Double[] inputs)
        {
            var actual = FeedForward(inputs).Output;
            var difference = actual - expected;

            foreach(var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for(Int32 i = Layers.Count - 2; i >= 0; --i)
            {
                var layer = Layers[i];
                var layerPrev = Layers[i + 1];
                for(Int32 j = 0; j < layer.NeuronsCount; ++j)
                {
                    var neuron = layer.Neurons[j];
                    for(Int32 k = 0; k < layerPrev.NeuronsCount; ++k)
                    {
                        var neuronPrev = layerPrev.Neurons[k];
                        var error = neuronPrev.Weights[i];
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }
            var result = difference * difference;

            return result;

        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (Int32 i = 1; i < Layers.Count; ++i)
            {
                var previousLayerSignals = Layers[i - 1].GetSignals();
                var currentLayer = Layers[i];

                foreach (var neuron in currentLayer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }
        private void SendSignalsToInputNeurons(Double[] inputSignals)
        {
            if (inputSignals.Length != Topology.InputCount)
                throw new Exception("Число входных сигналов должно быть равно числу нейронов входного слоя нейронной сети");
            for (Int32 i = 0; i < inputSignals.Length; ++i)
            {
                var signal = new Double[] { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }
    }
}
