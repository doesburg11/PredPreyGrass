using System.Collections.Generic;
using UnityEngine.Profiling;
using UnityEngine.Scripting.APIUpdating;

namespace UnityEngine.U2D.IK
{
    /// <summary>
    /// Component responsible for 2D Cyclic Coordinate Descent (CCD) IK.
    /// </summary>
    [MovedFrom("UnityEngine.Experimental.U2D.IK")]
    [Solver2DMenuAttribute("Chain (CCD)")]
    [IconAttribute(IconUtility.IconPath + "Animation.IKCCD.png")]
    public sealed class CCDSolver2D : Solver2D
    {
        const int k_MinIterations = 1;
        const float k_MinTolerance = 0.001f;
        const float k_MinVelocity = 0.01f;
        const float k_MaxVelocity = 1f;

        [SerializeField]
        IKChain2D m_Chain = new IKChain2D();

        [SerializeField]
        [Range(k_MinIterations, 50)]
        int m_Iterations = 10;

        [SerializeField]
        [Range(k_MinTolerance, 0.1f)]
        float m_Tolerance = 0.01f;

        [SerializeField]
        [Range(0f, 1f)]
        float m_Velocity = 0.5f;

        Vector3[] m_Positions;

        /// <summary>
        /// Get and set the solver's integration count.
        /// </summary>
        public int iterations
        {
            get => m_Iterations;
            set => m_Iterations = Mathf.Max(value, k_MinIterations);
        }

        /// <summary>
        /// Get and set target distance tolerance.
        /// </summary>
        public float tolerance
        {
            get => m_Tolerance;
            set => m_Tolerance = Mathf.Max(value, k_MinTolerance);
        }

        /// <summary>
        /// Get and Set the solver velocity.
        /// </summary>
        public float velocity
        {
            get => m_Velocity;
            set => m_Velocity = Mathf.Clamp01(value);
        }

        /// <summary>
        /// Returns the number of chains in the solver.
        /// </summary>
        /// <returns>Returns 1, because CCD Solver has only one chain.</returns>
        protected override int GetChainCount() => 1;

        /// <summary>
        /// Gets the chain in the solver at index.
        /// </summary>
        /// <param name="index">Index to query. Not used in this override.</param>
        /// <returns>Returns IKChain2D for the Solver.</returns>
        public override IKChain2D GetChain(int index) => m_Chain;

        /// <summary>
        /// Prepares the data required for updating the solver.
        /// </summary>
        protected override void DoPrepare()
        {
            if (m_Positions == null || m_Positions.Length != m_Chain.transformCount)
                m_Positions = new Vector3[m_Chain.transformCount];

            var root = m_Chain.rootTransform;
            for (var i = 0; i < m_Chain.transformCount; ++i)
                m_Positions[i] = root.TransformPoint((Vector2)root.InverseTransformPoint(m_Chain.transforms[i].position));
        }

        /// <summary>
        /// Updates the IK and sets the chain's transform positions.
        /// </summary>
        /// <param name="targetPositions">Target positions for the chain.</param>
        protected override void DoUpdateIK(List<Vector3> targetPositions)
        {
            Profiler.BeginSample(nameof(CCDSolver2D.DoUpdateIK));

            var root = m_Chain.rootTransform;
            var targetPosition = targetPositions[0];
            var targetLocalPosition2D = (Vector2)root.InverseTransformPoint(targetPosition);
            targetPosition = root.TransformPoint(targetLocalPosition2D);

            if (CCD2D.Solve(targetPosition, GetPlaneRootTransform().forward, iterations, tolerance, Mathf.Lerp(k_MinVelocity, k_MaxVelocity, m_Velocity), ref m_Positions))
            {
                for (var i = 0; i < m_Chain.transformCount - 1; ++i)
                {
                    var startLocalPosition = (Vector2)m_Chain.transforms[i + 1].localPosition;
                    var endLocalPosition = (Vector2)m_Chain.transforms[i].InverseTransformPoint(m_Positions[i + 1]);
                    m_Chain.transforms[i].localRotation *= Quaternion.AngleAxis(Vector2.SignedAngle(startLocalPosition, endLocalPosition), Vector3.forward);
                }
            }

            Profiler.EndSample();
        }
    }
}