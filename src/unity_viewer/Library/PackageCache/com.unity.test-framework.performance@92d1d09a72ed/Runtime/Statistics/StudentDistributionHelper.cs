using System;
using static System.Math;

namespace Unity.PerformanceTesting.Statistics
{
    static class StudentDistributionHelper
    {
        public static double InverseTwoTailedStudent(double p, double n)
        {
            var lower = 0.0;
            var upper = 1000.0;
            while (upper - lower > 1e-9)
            {
                var t = (lower + upper) / 2;
                var p2 = TwoTailedStudent(t, n);
                if (p2 < p)
                    upper = t;
                else
                    lower = t;
            }

            return (lower + upper) / 2;
        }

        /// <summary>
        /// ACM Algorithm 395: Student's t-distribution
        ///
        /// Evaluates the two-tail probability P(t|n) that t is exceeded
        /// in magnitude for Student's t-distribution with n degrees of freedom.
        ///
        /// http://dl.acm.org/citation.cfm?id=355599
        /// </summary>
        /// <param name="t">t-value, t > 0</param>
        /// <param name="n">Degree of freedom, n >= 1</param>
        /// <returns>2-tail p-value</returns>
        static double TwoTailedStudent(double t, double n)
        {
            if (t < 0)
                throw new ArgumentOutOfRangeException(nameof(t), "t should be >= 0");
            if (n < 1)
                throw new ArgumentOutOfRangeException(nameof(n), "n should be >= 1");

            t = t * t;
            var y = t / n;
            var b = y + 1.0;
            var nn = (int)Round(n);
            if (Abs(n - nn) > 1e-9 || n >= 20 || t < n && n > 200)
            {
                if (y > 1.0e-6)
                    y = Log(b);
                var a = n - 0.5;
                b = 48.0 * (a * a);
                y = a * y;
                y = (((((-0.4 * y - 3.3) * y - 24.0) * y - 85.5) / (0.8 * (y * y) + 100.0 + b) + y + 3.0) / b + 1.0) * Sqrt(y);
                return 2 * NormalDistributionHelper.Gauss(-y);
            }

            {
                double z = 1;

                double a;
                if (n < 20 && t < 4.0)
                {
                    y = Sqrt(y);
                    a = y;
                    if (nn == 1)
                        a = 0;
                }
                else
                {
                    a = Sqrt(b);
                    y = a * nn;
                    var j = 0;
                    while (Abs(a - z) > 0)
                    {
                        j += 2;
                        z = a;
                        y *= (j - 1) / (b * j);
                        a += y / (nn + j);
                    }

                    nn += 2;
                    z = 0;
                    y = 0;
                    a = -a;
                }

                while (true)
                {
                    nn -= 2;
                    if (nn > 1)
                        a = (nn - 1) / (b * nn) * a + y;
                    else
                        break;
                }

                a = nn == 0 ? a / Sqrt(b) : (Atan(y) + a / b) * 2 / PI;
                return z - a;
            }
        }
    }
}
