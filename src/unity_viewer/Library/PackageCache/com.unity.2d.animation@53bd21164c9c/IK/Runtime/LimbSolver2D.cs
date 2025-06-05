using System.Collections.Generic;
using UnityEngine.Scripting.APIUpdating;

namespace UnityEngine.U2D.IK
{
    /// <summary>
    /// Component responsible for 2D Limb IK.
    /// </summary>
    [MovedFrom("UnityEngine.Experimental.U2D.IK")]
    [Solver2DMenuAttribute("Limb")]
    [IconAttribute(IconUtility.IconPath + "Animation.IKLimb.png")]
    public sealed class LimbSolver2D : Solver2D
    {
        [SerializeField]
        IKChain2D m_Chain = new IKChain2D();

        [SerializeField]
        bool m_Flip;

        Vector3[] m_Positions = new Vector3[3];
        float[] m_Lengths = new float[2];
        float[] m_Angles = new float[2];

        /// <summary>
        /// Get and set the flip property.
        /// </summary>
        public bool flip
        {
            get => m_Flip;
            set => m_Flip = value;
        }

        /// <summary>
        /// Initializes the solver.
        /// </summary>
        protected override void DoInitialize()
        {
            m_Chain.transformCount = m_Chain.effector == null || IKUtility.GetAncestorCount(m_Chain.effector) < 2 ? 0 : 3;
            base.DoInitialize();
        }

        /// <summary>
        /// Returns the number of chains in the solver.
        /// </summary>
        /// <returns>Returns 1, because Limb Solver has only one chain.</returns>
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
            var lengths = m_Chain.lengths;
            m_Positions[0] = m_Chain.transforms[0].position;
            m_Positions[1] = m_Chain.transforms[1].position;
            m_Positions[2] = m_Chain.transforms[2].position;
            m_Lengths[0] = lengths[0];
            m_Lengths[1] = lengths[1];
        }

        /// <summary>
        /// Updates the IK and sets the chain's transform positions.
        /// </summary>
        /// <param name="targetPositions">List of target positions.</param>
        protected override void DoUpdateIK(List<Vector3> targetPositions)
        {
            var targetPosition = targetPositions[0];
            var upperLimb = m_Chain.transforms[0];
            var lowerLimb = m_Chain.transforms[1];
            var effector = m_Chain.effector;

            var targetLocalPosition2D = (Vector2)upperLimb.InverseTransformPoint(targetPosition);
            targetPosition = upperLimb.TransformPoint(targetLocalPosition2D);

            if (targetLocalPosition2D.sqrMagnitude > 0f && Limb.Solve(targetPosition, m_Lengths, m_Positions, ref m_Angles))
            {
                var upperLimbRotationAngle = Vector2.SignedAngle(Vector2.right, targetLocalPosition2D) + Vector2.SignedAngle(lowerLimb.localPosition, Vector2.right) + (flip ? -1f : 1f) * m_Angles[0];
                upperLimb.localRotation *= Quaternion.AngleAxis(upperLimbRotationAngle, Vector3.forward);

                var lowerLimbRotation = Vector2.SignedAngle(Vector2.right, lowerLimb.InverseTransformPoint(targetPosition)) + Vector2.SignedAngle(effector.localPosition, Vector2.right);
                m_Chain.transforms[1].localRotation *= Quaternion.AngleAxis(lowerLimbRotation, Vector3.forward);
            }
        }
    }
}