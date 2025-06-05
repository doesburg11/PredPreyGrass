using System;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteDragHandler : MouseManipulator
    {
        private readonly Action m_DragUpdated;
        private readonly Action m_DragPerformed;

        public TilePaletteDragHandler(Action dragUpdated, Action dragPerformed)
        {
            m_DragUpdated = dragUpdated;
            m_DragPerformed = dragPerformed;
            activators.Add(new ManipulatorActivationFilter { button = MouseButton.LeftMouse });
        }

        protected override void RegisterCallbacksOnTarget()
        {
            target.RegisterCallback<DragUpdatedEvent>(OnDragUpdate);
            target.RegisterCallback<DragPerformEvent>(OnDragPerformEvent);
        }

        protected override void UnregisterCallbacksFromTarget()
        {
            target.UnregisterCallback<DragUpdatedEvent>(OnDragUpdate);
            target.UnregisterCallback<DragPerformEvent>(OnDragPerformEvent);
        }

        private void OnDragUpdate(DragUpdatedEvent evt)
        {
            m_DragUpdated?.Invoke();
        }

        private void OnDragPerformEvent(DragPerformEvent evt)
        {
            m_DragPerformed?.Invoke();
        }
    }
}
