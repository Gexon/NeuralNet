using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class Neuron
    {
        public List<double> Weight { get; } // веса входящих в нейрон связей
        public List<double> Inputs { get; } // значения нейронов входящих связей
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }
        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weight = new List<double>();
            Inputs = new List<double>();
            InitWeightsRandomValues(inputCount);
        }

        private void InitWeightsRandomValues(int inputCount)
        {
            if (NeuronType == NeuronType.Input)
            {
                for (int i = 0; i < inputCount; i++)
                {
                    Weight.Add(1);
                    Inputs.Add(0);
                }
            }
            else
            {
                var rnd = new Random();
                for (int i = 0; i < inputCount; i++)
                {
                    Weight.Add(rnd.NextDouble());
                    Inputs.Add(0);
                }
            }           
        }

        /// <summary>
        /// Главный метод расчета нейрона
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double FeedForward(List<double> inputs)
        {
            //todo проверить входные параметры
            // сохраняем все значения нейронов связи с текущим нейроном
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weight[i];

            }

            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }

            return Output;
        }

        /// <summary>
        /// Нормализация выходного сигнала(сигмоид)
        /// </summary>
        /// <param name="x">Output</param>
        /// <returns>Нормализованное значение от 0 до 1</returns>
        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        /// <summary>
        /// Производная функции активации(для сигмоида)
        /// </summary>
        /// <param name="x">Output</param>
        /// <returns>Производная функции активации</returns>
        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }

        /// <summary>
        /// "Обучение" корректировка весов связей нейрона
        /// </summary>
        /// <param name="error">разница полученного значения нейрона и ожидаемого</param>
        /// <param name="learningRate">Скорость обучения(0,1)</param>
        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input) { return; } // входные нейроны не обучаем/не корректируем

            Delta = error * SigmoidDx(Output); //Delta - коэффициент корректировки весов

            // корректируем веса каждого входящего соединения
            for (int i = 0; i < Weight.Count; i++)
            {
                var weight = Weight[i]; // текущий вес входящего соединения
                var input = Inputs[i]; // выходное значение нейрона от входящего соединения
                var newWeihgt = weight - input * Delta * learningRate;
                Weight[i] = newWeihgt;
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
