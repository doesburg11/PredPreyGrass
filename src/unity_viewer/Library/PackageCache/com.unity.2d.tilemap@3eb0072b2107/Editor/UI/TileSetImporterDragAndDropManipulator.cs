using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;
using Object = UnityEngine.Object;

namespace UnityEditor.Tilemaps
{
    internal class TileSetImporterDragAndDropManipulator : Manipulator
    {
        const string k_DragReceiverClassName = "DragReceiver";

        event Action<IEnumerable<Texture2D>, bool> onDragPerform;
        Func<bool> m_CanStartDrag;

        bool m_IsDragging;
        bool m_IsChildDragged;

        bool isActiveDrag => m_IsDragging && !m_IsChildDragged;

        public TileSetImporterDragAndDropManipulator(Func<bool> canDragStart, Action<IEnumerable<Texture2D>, bool> dragPerform)
        {
            m_CanStartDrag = canDragStart;
            onDragPerform = dragPerform;
        }

        protected override void RegisterCallbacksOnTarget()
        {
            target.AddToClassList(k_DragReceiverClassName);

            target.RegisterCallback<DragEnterEvent>(OnDragEnter);
            target.RegisterCallback<DragPerformEvent>(OnDragPerform, TrickleDown.TrickleDown);
            target.RegisterCallback<DragUpdatedEvent>(OnDragUpdate, TrickleDown.TrickleDown);
            target.RegisterCallback<DragExitedEvent>(OnDragExit, TrickleDown.TrickleDown);
            target.RegisterCallback<DragLeaveEvent>(OnDragLeave);
        }

        protected override void UnregisterCallbacksFromTarget()
        {
            target.RemoveFromClassList(k_DragReceiverClassName);

            target.UnregisterCallback<DragEnterEvent>(OnDragEnter);
            target.UnregisterCallback<DragPerformEvent>(OnDragPerform);
            target.UnregisterCallback<DragUpdatedEvent>(OnDragUpdate);
            target.UnregisterCallback<DragExitedEvent>(OnDragExit);
            target.UnregisterCallback<DragLeaveEvent>(OnDragLeave);
        }

        void OnDragEnter(DragEnterEvent evt)
        {
            if (evt.currentTarget == evt.target)
                TryStartDrag();
        }

        void TryStartDrag()
        {
            if (m_IsDragging)
                return;

            var textures = RetrieveTextures(DragAndDrop.objectReferences);
            if (textures.Count == 0)
                return;

            if (!m_CanStartDrag())
                return;

            m_IsDragging = true;

            DragAndDrop.visualMode = DragAndDropVisualMode.Copy;
        }

        void StopDragging()
        {
            if (!m_IsDragging)
                return;

            m_IsDragging = false;

        }

        void OnDragUpdate(DragUpdatedEvent evt)
        {
            m_IsChildDragged = evt.currentTarget != evt.target;

            if (isActiveDrag)
                DragAndDrop.visualMode = DragAndDropVisualMode.Copy;
        }

        void OnDragExit(DragExitedEvent evt)
        {
            StopDragging();
        }

        void OnDragLeave(DragLeaveEvent evt)
        {
            StopDragging();
        }

        void OnDragPerform(DragPerformEvent evt)
        {
            if (!isActiveDrag)
                return;

            StopDragging();

            var textures = RetrieveTextures(DragAndDrop.objectReferences);
            if (textures.Count == 0)
                return;

            onDragPerform?.Invoke(textures, evt.altKey);
        }

        static HashSet<Texture2D> RetrieveTextures(Object[] objectReferences)
        {
            var textures = new HashSet<Texture2D>();
            foreach (var objectReference in objectReferences)
            {
                switch (objectReference)
                {
                    case Sprite sprite:
                    {
                        textures.Add(sprite.texture);
                        break;
                    }
                    case Texture2D texture2D:
                    {
                        textures.Add(texture2D);
                        break;
                    }
                }
            }
            return textures;
        }
    }
}
