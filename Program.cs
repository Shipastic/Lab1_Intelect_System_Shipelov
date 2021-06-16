using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lab1_Intelect_System_Shipelov
{
    enum People { 
    Игорь,
    Саша,
    Настя,
    Паша,
    Гена
    }
    class Vector
    {
        public double[] v; // значения вектора
        public int n; // длина вектора

        // конструктор из длины
        public Vector(int n)
        {
            this.n = n; // копируем длину
            v = new double[n]; // создаём массив
        }

        // создание вектора из вещественных значений
        public Vector(params double[] values)
        {
            n = values.Length;
            v = new double[n];

            for (int i = 0; i < n; i++)
                v[i] = values[i];
        }

        // обращение по индексу
        public double this[int i]
        {
            get { return v[i]; } // получение значение
            set { v[i] = value; } // изменение значения
        }
    }

    class Matrix
    {
        double[][] v; // значения матрицы
        public int n, m; // количество строк и столбцов

        // создание матрицы заданного размера и заполнение случайными числами из интервала (-0.5, 0.5)
        public Matrix(int n, int m, Random random)
        {
            this.n = n;
            this.m = m;

            v = new double[n][];

            for (int i = 0; i < n; i++)
            {
                v[i] = new double[m];

                for (int j = 0; j < m; j++)
                    v[i][j] = random.NextDouble() - 0.5; // заполняем случайными числами
            }
        }

        // обращение по индексу
        public double this[int i, int j]
        {
            get { return v[i][j]; } // получение значения
            set { v[i][j] = value; } // изменение значения
        }
    }

    class Network
    {
        struct LayerT
        {
            public Vector x; // вход слоя
            public Vector z; // активированный выход слоя
            public Vector df; // производная функции активации слоя
        }

        Matrix[] weights; // матрицы весов слоя
        LayerT[] L; // значения на каждом слое
        Vector[] deltas; // дельты ошибки на каждом слое

        int layersN; // число слоёв

        public Network(int[] sizes)
        {
            Random random = new Random(DateTime.Now.Millisecond); // создаём генератор случайных чисел

            layersN = sizes.Length - 1; // запоминаем число слоёв

            weights = new Matrix[layersN]; // создаём массив матриц весовых коэффициентов
            L = new LayerT[layersN]; // создаём массив значений на каждом слое
            deltas = new Vector[layersN]; // создаём массив для дельт

            for (int k = 1; k < sizes.Length; k++)
            {
                weights[k - 1] = new Matrix(sizes[k], sizes[k - 1], random); // создаём матрицу весовых коэффициентов

                L[k - 1].x = new Vector(sizes[k - 1]); // создаём вектор для входа слоя
                L[k - 1].z = new Vector(sizes[k]); // создаём вектор для выхода слоя
                L[k - 1].df = new Vector(sizes[k]); // создаём вектор для производной слоя

                deltas[k - 1] = new Vector(sizes[k]); // создаём вектор для дельт
            }
        }

        // прямое распространение
        public Vector Forward(Vector input)
        {
            for (int k = 0; k < layersN; k++)
            {
                if (k == 0)
                {
                    for (int i = 0; i < input.n; i++)
                        L[k].x[i] = input[i];
                }
                else
                {
                    for (int i = 0; i < L[k - 1].z.n; i++)
                        L[k].x[i] = L[k - 1].z[i];
                }

                for (int i = 0; i < weights[k].n; i++)
                {
                    double y = 0;

                    for (int j = 0; j < weights[k].m; j++)
                        y += weights[k][i, j] * L[k].x[j];

                    // активация с помощью сигмоидальной функции
                    L[k].z[i] = 1 / (1 + Math.Exp(-y));
                    L[k].df[i] = L[k].z[i] * (1 - L[k].z[i]);
                }
            }

            return L[layersN - 1].z; // возвращаем результат
        }
        // обратное распространение
        void Backward(Vector output, ref double error)
        {
            int last = layersN - 1;

            error = 0; // обнуляем ошибку

            for (int i = 0; i < output.n; i++)
            {
                double e = L[last].z[i] - output[i]; // находим разность значений векторов

                deltas[last][i] = e * L[last].df[i]; // запоминаем дельту
                error += e * e / 2; // прибавляем к ошибке половину квадрата значения
            }

            // вычисляем каждую предудущю дельту на основе текущей с помощью умножения на транспонированную матрицу
            for (int k = last; k > 0; k--)
            {
                for (int i = 0; i < weights[k].m; i++)
                {
                    deltas[k - 1][i] = 0;

                    for (int j = 0; j < weights[k].n; j++)
                        deltas[k - 1][i] += weights[k][j, i] * deltas[k][j];

                    deltas[k - 1][i] *= L[k - 1].df[i]; // умножаем получаемое значение на производную предыдущего слоя
                }
            }
        }
        // обновление весовых коэффициентов, alpha - скорость обучения
        void UpdateWeights(double alpha)
        {
            for (int k = 0; k < layersN; k++)
            {
                for (int i = 0; i < weights[k].n; i++)
                {
                    for (int j = 0; j < weights[k].m; j++)
                    {
                        weights[k][i, j] -= alpha * deltas[k][i] * L[k].x[j];
                    }
                }
            }
        }
        public void Train(Vector[] X, Vector[] Y, double alpha, double eps, int epochs)
        {
            int epoch = 1; // номер эпохи

            double error; // ошибка эпохи

            do
            {
                error = 0; // обнуляем ошибку

                // проходимся по всем элементам обучающего множества
                for (int i = 0; i < X.Length; i++)
                {
                    Forward(X[i]); // прямое распространение сигнала
                    Backward(Y[i], ref error); // обратное распространение ошибки
                    UpdateWeights(alpha); // обновление весовых коэффициентов
                }

                Console.WriteLine("Эпоха: {0}, Величина ошибки: {1}", epoch, error); // выводим в консоль номер эпохи и величину ошибку

                epoch++; // увеличиваем номер эпохи
            } while (epoch <= epochs && error > eps);
            Console.WriteLine("=========================================================================================================");
        }
    }


    class Program
    {
        static void Main(string[] args)
        {
            People people = new People();
            // массив входных обучающих векторов
            Vector[] X = 
                {
                new Vector(1, 1, 0 ),
                new Vector(1, 0, 1 ),
                new Vector(1, 1, 1 ),
                new Vector(0, 1, 1 )
                };

            // массив выходных обучающих векторов
            Vector[] Y = 
                {
                new Vector(1.0), 
                new Vector(0.0), 
                new Vector(1.0), 
                new Vector(0.0) 
                };

            Network network = new Network(new int[] { 3, 1, 1 }); // создаём сеть с тремя входами, тремя нейронами в скрытом слое и одним выходом
            network.Train(X, Y, 0.1, 1e-7, 100000); // запускаем обучение сети 
            for (int i = 0; i < 4; i++)
            {
                Vector output = network.Forward(X[i]);
                if (output[0] >= 0.4)
                {
                    Console.WriteLine("{3} : {0} {1} {2} | Работает по специальности: Да ",
                                    X[i][0], X[i][1], X[i][2], people++);
                }
                else {

                    Console.WriteLine("{3} : {0} {1} {2} | Работает по специальности: Нет",
                                        X[i][0], X[i][1], X[i][2], people++);
                }
            }
            Vector[] newVector = { new Vector(1, 0, 0) };
            Vector newOutput = network.Forward(newVector[0]);
            
            Console.WriteLine("=========================================================================================================");

            if (newOutput[0] >= 0.6)
            {
                Console.WriteLine("Гена : {0} {1} {2} |  Работает по специальности: Да", newVector[0][0], newVector[0][1], newVector[0][2]);
            }
            else
            {
                Console.WriteLine("Гена : {0} {1} {2} |  Работает по специальности: Нет", newVector[0][0], newVector[0][1], newVector[0][2]);
            }
            Console.ReadKey();
        }
    }
}
