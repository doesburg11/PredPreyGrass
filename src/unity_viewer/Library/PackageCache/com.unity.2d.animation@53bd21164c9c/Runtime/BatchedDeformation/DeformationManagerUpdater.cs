using Unity.Profiling;

namespace UnityEngine.U2D.Animation
{
    [AddComponentMenu("")]
    [DefaultExecutionOrder(UpdateOrder.spriteSkinUpdateOrder)]
    [ExecuteInEditMode]
    internal class DeformationManagerUpdater : MonoBehaviour
    {
        public System.Action<GameObject> onDestroyingComponent { get; set; }

        ProfilerMarker m_ProfilerMarker = new ProfilerMarker("DeformationManager.LateUpdate");

        void OnDestroy() => onDestroyingComponent?.Invoke(gameObject);

        void LateUpdate()
        {
            if (DeformationManager.instance.helperGameObject != gameObject)
            {
                GameObject.DestroyImmediate(gameObject);
                return;
            }

            m_ProfilerMarker.Begin();
            DeformationManager.instance.Update();
            m_ProfilerMarker.End();
        }
    }
}
