using System;
using System.Collections.Generic;
using UnityEngine.Scripting.APIUpdating;
using UnityEngine.Serialization;
using UnityEngine.U2D.Common;

namespace UnityEngine.U2D.IK
{
    /// <summary>
    /// Abstract class for implementing a 2D IK Solver.
    /// </summary>
    [MovedFrom("UnityEngine.Experimental.U2D.IK")]
    public abstract class Solver2D : MonoBehaviour, IPreviewable
    {
        [SerializeField]
        bool m_ConstrainRotation = true;

        [FormerlySerializedAs("m_RestoreDefaultPose")]
        [SerializeField]
        bool m_SolveFromDefaultPose = true;

        [SerializeField]
        [Range(0f, 1f)]
        float m_Weight = 1f;

        Plane m_Plane;
        List<Vector3> m_TargetPositions = new List<Vector3>();

        /// <summary>
        /// Used to evaluate if Solver2D needs to be updated.
        /// </summary>
        float m_LastFinalWeight;

        /// <summary>
        /// Returns the number of IKChain2D in the solver.
        /// </summary>
        public int chainCount => GetChainCount();

        /// <summary>
        /// Gets and sets the rotation constrain property.
        /// </summary>
        public bool constrainRotation
        {
            get => m_ConstrainRotation;
            set => m_ConstrainRotation = value;
        }

        /// <summary>
        /// Get and set restoring default pose before the update.
        /// </summary>
        public bool solveFromDefaultPose
        {
            get => m_SolveFromDefaultPose;
            set => m_SolveFromDefaultPose = value;
        }

        /// <summary>
        /// Returns true if the Solver2D is in a valid state.
        /// </summary>
        public bool isValid => Validate();

        /// <summary>
        /// Returns true if all chains in the Solver have a target.
        /// </summary>
        public bool allChainsHaveTargets => HasTargets();

        /// <summary>
        /// Get and Set Solver weights.
        /// </summary>
        public float weight
        {
            get => m_Weight;
            set => m_Weight = Mathf.Clamp01(value);
        }

        /// <summary>
        /// Validate new values set from the Inspector.
        /// </summary>
        protected virtual void OnValidate()
        {
            m_Weight = Mathf.Clamp01(m_Weight);
        }

        bool Validate()
        {
            for (var i = 0; i < GetChainCount(); ++i)
            {
                var chain = GetChain(i);
                if (!chain.isValid)
                    return false;
            }

            return DoValidate();
        }

        bool HasTargets()
        {
            for (var i = 0; i < GetChainCount(); ++i)
            {
                var chain = GetChain(i);
                if (chain.target == null)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Initializes the solver.
        /// </summary>
        public void Initialize()
        {
            DoInitialize();

            for (var i = 0; i < GetChainCount(); ++i)
            {
                var chain = GetChain(i);
                chain.Initialize();
            }
        }

        void Prepare()
        {
            var rootTransform = GetPlaneRootTransform();
            if (rootTransform != null)
            {
                m_Plane.normal = rootTransform.forward;
                m_Plane.distance = -Vector3.Dot(m_Plane.normal, rootTransform.position);
            }

            for (var i = 0; i < GetChainCount(); ++i)
            {
                var chain = GetChain(i);
                var constrainTargetRotation = constrainRotation && chain.target != null;

                if (m_SolveFromDefaultPose)
                    chain.RestoreDefaultPose(constrainTargetRotation);
            }

            DoPrepare();
        }

        void PrepareEffectorPositions()
        {
            m_TargetPositions.Clear();

            for (var i = 0; i < GetChainCount(); ++i)
            {
                var chain = GetChain(i);

                if (chain.target)
                    m_TargetPositions.Add(chain.target.position);
            }
        }

        /// <summary>
        /// Perform the Solver IK update.
        /// </summary>
        /// <param name="globalWeight">Weight for position solving.</param>
        public void UpdateIK(float globalWeight)
        {
            if (allChainsHaveTargets)
            {
                PrepareEffectorPositions();
                UpdateIK(m_TargetPositions, globalWeight);
            }
        }

        /// <summary>
        /// Perform the Solver IK update with specified target positions.
        /// </summary>
        /// <param name="targetPositions">Target positions.</param>
        /// <param name="globalWeight">Weight for position solving.</param>
        public void UpdateIK(List<Vector3> targetPositions, float globalWeight)
        {
            if (targetPositions.Count != chainCount)
                return;

            var finalWeight = globalWeight * weight;
            var weightValueChanged = Math.Abs(finalWeight - m_LastFinalWeight) > 0.0001f;
            m_LastFinalWeight = finalWeight;

            if (finalWeight == 0f && !weightValueChanged)
                return;

            if (!isValid)
                return;

            if (finalWeight < 1f)
                StoreLocalRotations();

            Prepare();

            DoUpdateIK(targetPositions);

            if (constrainRotation)
            {
                for (var i = 0; i < GetChainCount(); ++i)
                {
                    var chain = GetChain(i);

                    if (chain.target)
                        chain.effector.rotation = chain.target.rotation;
                }
            }

            if (finalWeight < 1f)
                BlendFkToIk(finalWeight);
        }

        void StoreLocalRotations()
        {
            for (var i = 0; i < GetChainCount(); ++i)
            {
                var chain = GetChain(i);
                chain.StoreLocalRotations();
            }
        }

        void BlendFkToIk(float finalWeight)
        {
            for (var i = 0; i < GetChainCount(); ++i)
            {
                var chain = GetChain(i);
                var constrainTargetRotation = constrainRotation && chain.target != null;
                chain.BlendFkToIk(finalWeight, constrainTargetRotation);
            }
        }

        /// <summary>
        /// Override to return the IKChain2D at the given index.
        /// </summary>
        /// <param name="index">Index of the IKChain2D.</param>
        /// <returns>A chain at a given index.</returns>
        public abstract IKChain2D GetChain(int index);

        /// <summary>
        /// Override to return the number of chains in the Solver.
        /// </summary>
        /// <returns>Number of chains in the solver.</returns>
        protected abstract int GetChainCount();

        /// <summary>
        /// Override to perform Solver IK update.
        /// </summary>
        /// <param name="targetPositions">Target position for the chain.</param>
        protected abstract void DoUpdateIK(List<Vector3> targetPositions);

        /// <summary>
        /// Override to perform custom validation.
        /// </summary>
        /// <returns>Returns true if the Solver is in a valid state. False otherwise.</returns>
        protected virtual bool DoValidate() => true;

        /// <summary>
        /// Override to initialize the solver.
        /// </summary>
        protected virtual void DoInitialize() { }

        /// <summary>
        /// Override to prepare the solver for update.
        /// </summary>
        protected virtual void DoPrepare() { }

        /// <summary>
        /// Override to return the root transform of the Solver. The default implementation returns the root transform of the first chain.
        /// </summary>
        /// <returns>Transform representing the root.</returns>
        protected virtual Transform GetPlaneRootTransform()
        {
            return chainCount > 0 ? GetChain(0).rootTransform : null;
        }

        /// <summary>
        /// Convert a world position coordinate to the solver's plane space.
        /// </summary>
        /// <param name="worldPosition">Vector3 representing world position</param>
        /// <returns>Converted position in solver's plane</returns>
        protected Vector3 GetPointOnSolverPlane(Vector3 worldPosition)
        {
            return GetPlaneRootTransform().InverseTransformPoint(m_Plane.ClosestPointOnPlane(worldPosition));
        }

        /// <summary>
        /// Convert a position from solver's plane to world coordinates.
        /// </summary>
        /// <param name="planePoint">Vector3 representing a position in the Solver's plane.</param>
        /// <returns>Converted position to world coordinates.</returns>
        protected Vector3 GetWorldPositionFromSolverPlanePoint(Vector2 planePoint)
        {
            return GetPlaneRootTransform().TransformPoint(planePoint);
        }

        /// <summary>
        /// Empty method. Implemented for the IPreviewable interface.
        /// </summary>
        public void OnPreviewUpdate() { }
    }
}