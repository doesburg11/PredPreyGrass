using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
    /// <summary>
    /// Manipulator that allows dragging from a container component.
    /// </summary>
    internal class Dragger : PointerManipulator
    {
        /// <summary>
        /// The threshold in pixels after which a drag will start.
        /// </summary>
        public float dragThreshold { get; set; } = 8f;

        /// <summary>
        /// Whether the drag is currently active.
        /// </summary>
        public bool isActive { get; private set; }

        /// <summary>
        /// Delegate that will be called when a drag should be accepted during <see cref="PointerDownEvent"/>.
        /// </summary>
        /// <remarks>
        /// If this delegate is not set (or set to null), the drag will be accepted by default.
        /// </remarks>
        public Func<Vector2, bool> acceptStartDrag { get; set; }

        /// <summary>
        /// Delegate that will be called when a drag should be accepted when the dragger wants to become
        /// active during <see cref="PointerMoveEvent"/>.
        /// </summary>
        /// <remarks>
        /// If this delegate is not set (or set to null), the drag will be accepted by default.
        /// </remarks>
        public Func<bool> acceptDrag { get; set; }

        int m_PointerId = -1;

        Vector3 m_StartPosition;

        event Action<PointerMoveEvent> onDragStarted;
        event Action<PointerMoveEvent> onDragging;
        event Action<PointerUpEvent> onDragEnded;
        event Action onDragCanceled;

        /// <summary>
        /// Creates a new <see cref="Dragger"/> instance.
        /// </summary>
        /// <param name="dragStarted"> The event that will be fired when a drag starts. </param>
        /// <param name="dragging"> The event that will be fired when a drag is in progress. </param>
        /// <param name="dragEnded"> The event that will be fired when a drag ends. </param>
        /// <param name="dragCanceled"> The event that will be fired when a drag is cancelled. </param>
        public Dragger(Action<PointerMoveEvent> dragStarted, Action<PointerMoveEvent> dragging,
            Action<PointerUpEvent> dragEnded, Action dragCanceled)
        {
            onDragStarted = dragStarted;
            onDragging = dragging;
            onDragEnded = dragEnded;
            onDragCanceled = dragCanceled;
        }

        /// <inheritdoc cref="Manipulator.RegisterCallbacksOnTarget"/>
        protected override void RegisterCallbacksOnTarget()
        {
            activators.Add(new ManipulatorActivationFilter { button = MouseButton.LeftMouse });
            target.RegisterCallback<PointerDownEvent>(OnPointerDown);
            target.RegisterCallback<PointerMoveEvent>(OnPointerMove);
            target.RegisterCallback<PointerUpEvent>(OnPointerUp);
            target.RegisterCallback<PointerCancelEvent>(OnPointerCancel);
            target.RegisterCallback<PointerCaptureOutEvent>(OnPointerCaptureOut);
            target.RegisterCallback<KeyDownEvent>(OnKeyDown);
        }

        /// <inheritdoc cref="Manipulator.UnregisterCallbacksFromTarget"/>
        protected override void UnregisterCallbacksFromTarget()
        {
            activators.Clear();
            target.UnregisterCallback<PointerDownEvent>(OnPointerDown);
            target.UnregisterCallback<PointerMoveEvent>(OnPointerMove);
            target.UnregisterCallback<PointerUpEvent>(OnPointerUp);
            target.UnregisterCallback<PointerCancelEvent>(OnPointerCancel);
            target.UnregisterCallback<PointerCaptureOutEvent>(OnPointerCaptureOut);
            target.UnregisterCallback<KeyDownEvent>(OnKeyDown);
        }

        void OnPointerDown(PointerDownEvent evt)
        {
            isActive = false;
            m_PointerId = -1;

            if (CanStartManipulation(evt) && (acceptStartDrag == null || acceptStartDrag.Invoke(evt.position)))
            {
                m_StartPosition = evt.position;
                m_PointerId = evt.pointerId;
            }
        }

        void OnPointerMove(PointerMoveEvent evt)
        {
            if (m_PointerId != evt.pointerId)
                return;

            if (m_PointerId != -1)
            {
                if (!isActive)
                {
                    var delta = evt.position - m_StartPosition;
                    if (delta.sqrMagnitude > dragThreshold * dragThreshold && (acceptDrag == null || acceptDrag.Invoke()))
                    {
                        if (!target.HasPointerCapture(evt.pointerId))
                        {
                            target.CapturePointer(evt.pointerId);
#if !UNITY_2023_1_OR_NEWER
                            if (evt.pointerId == PointerId.mousePointerId)
                                target.CaptureMouse();
#endif
                        }

                        isActive = true;
                        onDragStarted?.Invoke(evt);
                    }
                }
                else
                {
                    if (!target.HasPointerCapture(evt.pointerId))
                    {
                        target.CapturePointer(evt.pointerId);
#if !UNITY_2023_1_OR_NEWER
                        if (evt.pointerId == PointerId.mousePointerId)
                            target.CaptureMouse();
#endif
                    }

                    onDragging?.Invoke(evt);
                }
            }
        }

        void OnPointerUp(PointerUpEvent evt)
        {
            if (m_PointerId != evt.pointerId)
                return;

            if (target.HasPointerCapture(evt.pointerId))
                target.ReleasePointer(evt.pointerId);

            if (isActive)
                onDragEnded?.Invoke(evt);

            isActive = false;
            m_PointerId = -1;
        }

        void OnPointerCancel(PointerCancelEvent evt)
        {
            if (evt.pointerId == m_PointerId)
                Cancel();
        }

        void OnPointerCaptureOut(PointerCaptureOutEvent evt)
        {
            if (evt.pointerId == m_PointerId)
                Cancel();
        }

        void OnKeyDown(KeyDownEvent evt)
        {
            if (m_PointerId == -1)
                return;

            if (evt.keyCode == KeyCode.Escape)
            {

                evt.StopImmediatePropagation();
                Cancel();
            }
        }

        /// <summary>
        /// Cancels the drag.
        /// </summary>
        public void Cancel()
        {
            if (target.HasPointerCapture(m_PointerId))
                target.ReleasePointer(m_PointerId);

            if (isActive)
            {
                isActive = false;
                onDragCanceled?.Invoke();
            }

            m_PointerId = -1;
        }
    }
}
