using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; }
        public Topology Topology { get; }
        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();
            CreateInputLayer();
            CreateHiddenLayer();
            CreateOutputLayer();
        }

        public Neuron FeedForward(params double[] inputSignals)
        {
            //todo проверить кол-во входных сигналов к количеству входных нейронов нашей сети
            SendSignalsToInputNeurons(inputSignals); //ситаем первый входной слой
            FeedForwardAllLayersAfterInput(); //считаем все остальные слои
            //конечный результат 
            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        /// <summary>
        /// Обучение, корректировка весов.
        /// Каждое прохождение всего набора - это 1 эпоха, корерктирует веса всей сети равным количеству вариантов набора(16 раз).
        /// В данном тесте 1000 эпох, 1000*16 раз корректировалась сеть(16 000 раз).
        /// </summary>
        /// <param name="dataset">набор входных данных(16 различных вариантов входных данных)</param>
        /// <param name="epoch">количество проходов сети(1000), используя набор входных данных</param>
        /// <returns>возвращаем среднее значение ошибки, после прохождения всех эпох</returns>
        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            var error = 0.0;

            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);                    
                    error += Backpropagation(output, input);
                }
            }

            var result = error / epoch;
            return result;
        }

        // получаем строку с двумерного массива
        public static double[] GetRow(double[,] matrix, int row)
        {
            var colums = matrix.GetLength(1);
            var result = new double[colums];
            for (int i = 0; i < colums; ++i)
            {
                result[i] = matrix[row, i];
            }
            return result;
        }

        /// <summary>
        /// Скалирование датасета [0..1]
        /// </summary>
        /// <param name="inputs">Датасет</param>
        /// <returns>Отмасштабированные данные</returns>
        public double[,] Scalling(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];

                // ищем мин и макс
                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, column];
                    if (item < min)
                    {
                        min = item;
                    }
                    if (item > max)
                    {
                        max = item;
                    }
                }

                var divider = max - min;
                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - min) / divider;
                }
            }

            return result;
        }

        /// <summary>
        /// Нормализация датасета [-1..1] или [-2..2] =)
        /// </summary>
        /// <param name="inputs">Датасет</param>
        /// <returns>Нормализованные данные</returns>
        public double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                // вычисляем среднее значение сигнала нейрона
                var sum = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }
                var average = sum / inputs.GetLength(0);

                // стандартное квадратичное отклонение
                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, column] - average), 2);
                }
                var standardError = Math.Sqrt(error / inputs.GetLength(0));

                // нрмализованное значение
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standardError;
                }
            }

            return result;
        }

        /// <summary>
        /// Реализация метода "Обратного распространения ошибки"
        /// </summary>
        /// <param name="expected">Ожидаемое значение</param>
        /// <param name="inputs">Входные данные, значения входных нейронов</param>
        /// <returns></returns>
        private double Backpropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output; // полученный результат с выходного нейрона после прохождения всей сети
            var difference = actual - expected; // разница значений(Z), фактического значения на выходе нейрона и ожидаемого значения

            // корректировка весов(входящих связей) выходного(итогово, последнего) нейрона
            // difference в данном примере для 1 выходного нейрона
            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate); // корректировка весов нейрона(в выходном/последней слое)
            }

            for (var j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weight[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            var result = difference * difference;
            return result;
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        /// <summary>
        /// Первый слой, слой входных нейронов
        /// </summary>
        /// <param name="inputSignals">Входные сигналы сети</param>
        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double> { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];
                neuron.FeedForward(signal);
            }
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        private void CreateHiddenLayer()
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }


    }
}
