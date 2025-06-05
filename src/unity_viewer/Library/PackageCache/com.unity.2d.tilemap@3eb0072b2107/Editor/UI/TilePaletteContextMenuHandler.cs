using System;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteContextMenuHandler : MouseManipulator
    {
        private readonly Action m_ContextClick;

        public TilePaletteContextMenuHandler(Action contextClick)
        {
            m_ContextClick = contextClick;
            activators.Add(new ManipulatorActivationFilter { button = MouseButton.RightMouse });
        }

        protected override void RegisterCallbacksOnTarget()
        {
            target.RegisterCallback<ContextClickEvent>(OnContextClick);
        }

        protected override void UnregisterCallbacksFromTarget()
        {
            target.UnregisterCallback<ContextClickEvent>(OnContextClick);
        }

        private void OnContextClick(ContextClickEvent evt)
        {
            m_ContextClick?.Invoke();
        }
    }
}
