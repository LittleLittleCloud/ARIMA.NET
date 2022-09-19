using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ARIMANet
{
    public class ARMA
    {
        private readonly int p;
        private readonly int q;
        private readonly int m;
        private ITransformer regressor;
        private readonly MLContext context;
        private IEnumerable<float> prevX;
        private IEnumerable<float> prevZ;

        public ARMA(MLContext context, int p, int q)
        {
            this.p = p;
            this.q = q;
            this.m = Math.Max(p, q) * 2;
            this.context = context;
        }

        public virtual void Fit(IEnumerable<float> x)
        {
            // fit ar(m)
            // construct _X from X with size-m slide window
            var trainX = this.SlidingWindowView(x.SkipLast(1), this.m);
            var trainY = x.Skip(this.m);
            var schemaDefinition = SchemaDefinition.Create(typeof(RegressionInput));
            schemaDefinition["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, this.m);
            var inputs = Enumerable.Zip(trainX, trainY).Select(x => new RegressionInput
            {
                Features = x.First,
                Label = x.Second,
            });
            var trainData = this.context.Data.LoadFromEnumerable(inputs, schemaDefinition);

            Console.WriteLine($"start fitting ar({this.m})");
            var pipeline = this.context.Regression.Trainers.Ols();
            var arMmodel = pipeline.Fit(trainData);
            var evalD = arMmodel.Transform(trainData);
            var predictY = evalD.GetColumn<float>("Score").ToArray();
            Console.WriteLine($"finish fitting ar({this.m})");
            var metric = this.context.Regression.Evaluate(evalD);
            var zUp = Enumerable.Zip(trainY, predictY).Select(x => x.First - x.Second);
            Console.WriteLine($"rmse : {metric.RootMeanSquaredError}");
            var _z = this.SlidingWindowView(zUp, this.q).SkipLast(1);
            var _x = this.SlidingWindowView(x, this.p).TakeLast(_z.Count() + 1).SkipLast(1);
            var z = Enumerable.Zip(_x, _z).Select(x => x.First.Concat(x.Second).ToArray());
            var y = trainY.Skip(this.q);
            var arModelPipeline = this.context.Regression.Trainers.Ols();
            schemaDefinition["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, this.p + this.q);

            inputs = Enumerable.Zip(z, y).Select(x => new RegressionInput
            {
                Features = x.First,
                Label = x.Second,
            }).ToArray();
            var data = this.context.Data.LoadFromEnumerable(inputs, schemaDefinition);
            Console.WriteLine($"start fitting ar({this.p + this.q})");
            this.regressor = arModelPipeline.Fit(data);
            Console.WriteLine($"end fitting ar({this.p + this.q})");

            // evaluate
            var eval = this.regressor.Transform(data);
            var matric = this.context.Regression.Evaluate(eval);
            Console.WriteLine("rmse for ar(p+q)");
            Console.WriteLine(matric.RootMeanSquaredError);
            this.prevX = z.Last().Take(this.p);
            this.prevZ = z.Last().Skip(this.p);
        }

        public virtual IEnumerable<float> Predict(int n = 1)
        {
            var res = new List<float>();
            var prevX = this.prevX;
            var prevZ = this.prevZ;
            var schemaDefinition = SchemaDefinition.Create(typeof(RegressionInput));
            schemaDefinition["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, this.p + this.q);

            for (int i = 0; i!= n; ++i)
            {
                var input = new RegressionInput
                {
                    Features = prevX.Concat(prevZ).ToArray(),
                    Label = 0,
                };

                var data = this.context.Data.LoadFromEnumerable(new[] { input }, schemaDefinition);
                var output = this.regressor.Transform(data);
                var label = output.GetColumn<float>("Score").First();
                res.Add(label);

                if(this.p > 0)
                {
                    prevX = prevX.Skip(1).Append(label);
                }
                if(this.q > 0)
                {
                    prevZ = prevZ.Skip(1).Append(0);
                }
            }

            return res;
        }

        public virtual void Update(float x)
        {
            var predict = this.Predict(1);

            if (this.p > 0)
            {
                this.prevX = this.prevX.Skip(1).Append(x);
            }

            if(this.q > 0)
            {
                this.prevZ = this.prevZ.Skip(1).Append(x - predict.First());
            }
        }

        protected IEnumerable<float[]> SlidingWindowView(IEnumerable<float> x, int windowSize)
        {
            var _x = x.ToArray();
            for(int i = 0; i != _x.Count() - windowSize + 1; ++i)
            {
                yield return _x[i..(i+ windowSize)].ToArray();
            }
        }

        class RegressionInput
        {
            public float[] Features;

            [ColumnName("Label")]
            public float Label;
        }
    }
}
