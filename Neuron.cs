﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class Neuron
    {
        public List<double> Weight { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weight = new List<double>();

            for (int i = 0; i < inputCount; i++)
            {
                Weight.Add(1);

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

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weight[i];

            }
            Output = Sigmoid(sum);
            return Output;
        }

        /// <summary>
        /// Нормализация выходного сигнала
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
