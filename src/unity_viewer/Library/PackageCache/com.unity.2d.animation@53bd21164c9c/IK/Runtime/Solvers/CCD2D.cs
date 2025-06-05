using UnityEngine.Scripting.APIUpdating;

namespace UnityEngine.U2D.IK
{
    /// <summary>
    /// Utility for 2D based Cyclic Coordinate Descent (CCD) IK Solver.
    /// </summary>
    [MovedFrom("UnityEngine.Experimental.U2D.IK")]
    public static class CCD2D
    {
        /// <summary>
        /// Solve IK Chain based on CCD.
        /// </summary>
        /// <param name="targetPosition">Target position.</param>
        /// <param name="forward">Forward vector for solver.</param>
        /// <param name="solverLimit">Solver iteration count.</param>
        /// <param name="tolerance">Target position's tolerance.</param>
        /// <param name="velocity">Velocity towards target position.</param>
        /// <param name="positions">Chain positions.</param>
        /// <returns>Returns true if solver successfully completes within iteration limit. False otherwise.</returns>
        public static bool Solve(Vector3 targetPosition, Vector3 forward, int solverLimit, float tolerance, float velocity, ref Vector3[] positions)
        {
            var last = positions.Length - 1;
            var iterations = 0;
            var sqrTolerance = tolerance * tolerance;
            var sqrDistanceToTarget = (targetPosition - positions[last]).sqrMagnitude;
            while (sqrDistanceToTarget > sqrTolerance)
            {
                DoIteration(targetPosition, forward, last, velocity, ref positions);
                sqrDistanceToTarget = (targetPosition - positions[last]).sqrMagnitude;
                if (++iterations >= solverLimit)
                    break;
            }

            return iterations != 0;
        }

        static void DoIteration(Vector3 targetPosition, Vector3 forward, int last, float velocity, ref Vector3[] positions)
        {
            for (var i = last - 1; i >= 0; --i)
            {
                var toTarget = targetPosition - positions[i];
                var toLast = positions[last] - positions[i];

                var angle = Vector3.SignedAngle(toLast, toTarget, forward);
                angle = Mathf.Lerp(0f, angle, velocity);

                var deltaRotation = Quaternion.AngleAxis(angle, forward);
                for (var j = last; j > i; --j)
                    positions[j] = RotatePositionFrom(positions[j], positions[i], deltaRotation);
            }
        }

        static Vector3 RotatePositionFrom(Vector3 position, Vector3 pivot, Quaternion rotation)
        {
            var v = position - pivot;
            v = rotation * v;
            return pivot + v;
        }
    }
}