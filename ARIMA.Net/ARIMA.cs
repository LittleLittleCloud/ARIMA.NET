using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ARIMANet
{
    public class ARIMA : ARMA
    {
        private readonly ARMA model;
        private readonly int d;
        private IEnumerable<float> z;
        private float x0;

        public ARIMA(MLContext context, int p, int q, int d)
            : base(context, p, q)
        {
            if(d == 0)
            {
                this.d = d;
                this.model = new ARMA(context, p, q);
            }
            else
            {
                this.d = d;
                this.model = new ARIMA(context, p, q, d - 1);
            }
        }

        public override void Fit(IEnumerable<float> x)
        {
            if (this.d == 0)
            {
                this.z = x;
                this.model.Fit(x);
            }
            else
            {
                this.x0 = x.First();
                this.z = base.SlidingWindowView(x, 2).Select(x => x.Last() - x.First());
                this.model.Fit(this.z);
            }
        }

        public override IEnumerable<float> Predict(int n = 1)
        {
            if(this.d == 0)
            {
                return this.model.Predict(n);
            }
            else
            {
                var zTplus1 = this.model.Predict(n);
                var res = new List<float>();
                var sumZ = this.z.Sum() + this.x0;
                foreach(var z in zTplus1)
                {
                    res.Add(sumZ + z);
                    sumZ += z;
                }

                return res;
            }
        }

        public override void Update(float x)
        {
            if(this.d == 0)
            {
                this.model.Update(x);
            }
            else
            {
                var xT = this.z.Sum() + this.x0;
                var zTPlus1 = x - xT;
                this.z = this.z.Append(zTPlus1);
                this.model.Update(zTPlus1);
            }
        }
    }
}
