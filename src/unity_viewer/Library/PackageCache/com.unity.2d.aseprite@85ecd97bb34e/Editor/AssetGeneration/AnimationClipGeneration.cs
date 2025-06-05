using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    internal static class AnimationClipGeneration
    {
        const string k_RootName = "Root";
        const string k_CombinedEventName = "OnAnimationEvent";

        public static AnimationClip[] Generate(string assetName,
            IReadOnlyList<Sprite> sprites,
            AsepriteFile file,
            IReadOnlyList<Layer> layers,
            IReadOnlyList<Frame> frames,
            List<Tag> tags,
            Dictionary<int, GameObject> layerIdToGameObject,
            bool generateIndividualEvents)
        {
            var noOfFrames = file.noOfFrames;
            if (tags.Count == 0)
            {
                var tag = new Tag();
                tag.name = assetName + "_Clip";
                tag.fromFrame = 0;
                tag.toFrame = noOfFrames;

                tags.Add(tag);
            }

            var layersWithDisabledRenderer = new HashSet<Layer>();
            for (var i = 0; i < layers.Count; ++i)
            {
                if (DoesLayerDisableRenderer(layers[i], tags))
                    layersWithDisabledRenderer.Add(layers[i]);
            }
            var layersWithCustomSortingOrder = new HashSet<Layer>();
            for (var i = 0; i < layers.Count; ++i)
            {
                if (DoesLayerHaveCustomSorting(layers[i]))
                    layersWithCustomSortingOrder.Add(layers[i]);
            }

            var clips = new List<AnimationClip>(tags.Count);
            var animationNames = new HashSet<string>(tags.Count);
            for (var i = 0; i < tags.Count; ++i)
            {
                var clipName = tags[i].name;
                if (animationNames.Contains(clipName))
                {
                    var nameIndex = 0;
                    while (animationNames.Contains(clipName))
                        clipName = $"{tags[i].name}_{nameIndex++}";

                    Debug.LogWarning($"The animation clip name {tags[i].name} is already in use. Renaming to {clipName}.");
                }

                var clip = CreateClip(tags[i], clipName, layers, layersWithDisabledRenderer, layersWithCustomSortingOrder, frames, sprites, layerIdToGameObject, generateIndividualEvents);
                clips.Add(clip);
                animationNames.Add(clipName);
            }

            return clips.ToArray();
        }

        static bool DoesLayerDisableRenderer(Layer layer, IReadOnlyList<Tag> tags)
        {
            if (layer.layerType != LayerTypes.Normal)
                return false;

            var cells = layer.cells;
            var linkedCells = layer.linkedCells;

            for (var i = 0; i < tags.Count; ++i)
            {
                var tag = tags[i];
                for (var frameIndex = tag.fromFrame; frameIndex < tag.toFrame; ++frameIndex)
                {
                    var foundCell = false;
                    foreach (var cell in cells)
                    {
                        if (cell.frameIndex != frameIndex)
                            continue;
                        foundCell = true;
                        break;
                    }

                    if (foundCell)
                        continue;

                    foreach (var cell in linkedCells)
                    {
                        if (cell.frameIndex != frameIndex)
                            continue;
                        foundCell = true;
                        break;
                    }

                    if (!foundCell)
                        return true;
                }
            }
            return false;
        }

        static bool DoesLayerHaveCustomSorting(Layer layer)
        {
            if (layer.layerType != LayerTypes.Normal)
                return false;
            
            var cells = layer.cells;
            var linkedCells = layer.linkedCells;

            foreach(var cell in cells)
            {
                if (cell.additiveSortOrder != 0)
                    return true;
            }

            foreach (var linkedCell in linkedCells)
            {
                var cellIndex = cells.FindIndex(x => x.frameIndex == linkedCell.linkedToFrame);
                if (cellIndex == -1)
                    continue;

                var cell = cells[cellIndex];
                if (cell.additiveSortOrder != 0)
                    return true;
            }
            
            return false;            
        }

        static AnimationClip CreateClip(
            Tag tag, 
            string clipName, 
            IReadOnlyList<Layer> layers, 
            IReadOnlyCollection<Layer> layersWithDisabledRenderer, 
            IReadOnlyCollection<Layer> layersWithCustomSorting, 
            IReadOnlyList<Frame> frames, 
            IReadOnlyList<Sprite> sprites, 
            IReadOnlyDictionary<int, GameObject> layerIdToGameObject,
            bool generateIndividualEvents)
        {
            var animationClip = new AnimationClip()
            {
                name = clipName,
                frameRate = 100f
            };

            var clipSettings = new AnimationClipSettings();
            clipSettings.loopTime = tag.isRepeating;
            AnimationUtility.SetAnimationClipSettings(animationClip, clipSettings);

            for (var i = 0; i < layers.Count; ++i)
            {
                var layer = layers[i];
                if (layer.layerType != LayerTypes.Normal)
                    continue;

                var layerGo = layerIdToGameObject[layer.index];
                if (layerGo.GetComponent<SpriteRenderer>() == null)
                    continue;

                var doesLayerDisableRenderer = layersWithDisabledRenderer.Contains(layer);
                var doesLayerHaveCustomSorting = layersWithCustomSorting.Contains(layer);
                
                var layerTransform = layerGo.transform;
                var spriteKeyframes = new List<ObjectReferenceKeyframe>();

                var cells = layer.cells;
                var activeFrames = AddCellsToClip(cells, tag, sprites, frames, spriteKeyframes);

                var linkedCells = layer.linkedCells;
                activeFrames.UnionWith(AddLinkedCellsToClip(linkedCells, cells, tag, sprites, frames, spriteKeyframes));

                spriteKeyframes.Sort((x, y) => x.time.CompareTo(y.time));
                DuplicateLastFrame(spriteKeyframes, frames[tag.toFrame - 1], animationClip.frameRate);

                var path = GetTransformPath(layerTransform);
                var spriteBinding = EditorCurveBinding.PPtrCurve(path, typeof(SpriteRenderer), "m_Sprite");
                AnimationUtility.SetObjectReferenceCurve(animationClip, spriteBinding, spriteKeyframes.ToArray());

                AddEnabledKeyframes(layerTransform, tag, frames, doesLayerDisableRenderer, activeFrames, animationClip);
                AddSortOrderKeyframes(layerTransform, layer, tag, frames, cells, doesLayerHaveCustomSorting, animationClip);
                AddAnimationEvents(tag, frames, animationClip, generateIndividualEvents);
            }

            return animationClip;
        }

        static HashSet<int> AddCellsToClip(IReadOnlyList<Cell> cells, Tag tag, IReadOnlyList<Sprite> sprites, IReadOnlyList<Frame> frames, List<ObjectReferenceKeyframe> keyFrames)
        {
            var activeFrames = new HashSet<int>();
            var startTime = GetTimeFromFrame(frames, tag.fromFrame);
            for (var i = 0; i < cells.Count; ++i)
            {
                var cell = cells[i];
                if (cell.frameIndex < tag.fromFrame ||
                    cell.frameIndex >= tag.toFrame)
                    continue;

                var sprite = sprites.Find(x => x.GetSpriteID() == cell.spriteId);
                if (sprite == null)
                    continue;

                var keyframe = new ObjectReferenceKeyframe();
                var time = GetTimeFromFrame(frames, cell.frameIndex);
                keyframe.time = time - startTime;
                keyframe.value = sprite;
                keyFrames.Add(keyframe);

                activeFrames.Add(cell.frameIndex);
            }
            return activeFrames;
        }

        static HashSet<int> AddLinkedCellsToClip(IReadOnlyList<LinkedCell> linkedCells, IReadOnlyList<Cell> cells, Tag tag, IReadOnlyList<Sprite> sprites, IReadOnlyList<Frame> frames, List<ObjectReferenceKeyframe> keyFrames)
        {
            var activeFrames = new HashSet<int>();
            var startTime = GetTimeFromFrame(frames, tag.fromFrame);
            for (var i = 0; i < linkedCells.Count; ++i)
            {
                var linkedCell = linkedCells[i];
                if (linkedCell.frameIndex < tag.fromFrame ||
                    linkedCell.frameIndex >= tag.toFrame)
                    continue;
                
                var cellIndex = cells.FindIndex(x => x.frameIndex == linkedCell.linkedToFrame);
                if (cellIndex == -1)
                    continue;

                var cell = cells[cellIndex];
                var sprite = sprites.Find(x => x.GetSpriteID() == cell.spriteId);
                if (sprite == null)
                    continue;

                var keyframe = new ObjectReferenceKeyframe();
                var time = GetTimeFromFrame(frames, linkedCell.frameIndex);
                keyframe.time = time - startTime;
                keyframe.value = sprite;
                keyFrames.Add(keyframe);

                activeFrames.Add(linkedCell.frameIndex);
            }
            return activeFrames;
        }

        static void DuplicateLastFrame(List<ObjectReferenceKeyframe> keyFrames, Frame lastFrame, float frameRate)
        {
            if (keyFrames.Count == 0)
                return;

            var frameTime = 1f / frameRate;

            var lastKeyFrame = keyFrames[^1];
            var duplicatedFrame = new ObjectReferenceKeyframe();

            var time = lastKeyFrame.time + MsToSeconds(lastFrame.duration);
            // We remove one AnimationClip frame, since the animation system will automatically add one frame at the end.
            time -= frameTime;
            duplicatedFrame.time = time;
            duplicatedFrame.value = lastKeyFrame.value;
            keyFrames.Add(duplicatedFrame);
        }

        static string GetTransformPath(Transform transform)
        {
            var path = transform.name;
            if (transform.name == k_RootName)
                return "";
            if (transform.parent.name == k_RootName)
                return path;

            var parentPath = GetTransformPath(transform.parent) + "/";
            path = path.Insert(0, parentPath);
            return path;
        }

        static void AddEnabledKeyframes(Transform layerTransform, Tag tag, IReadOnlyList<Frame> frames, bool doesLayerDisableRenderer, IReadOnlyCollection<int> activeFrames, AnimationClip animationClip)
        {
            if (activeFrames.Count == tag.noOfFrames && !doesLayerDisableRenderer)
                return;

            var path = GetTransformPath(layerTransform);
            var enabledBinding = EditorCurveBinding.FloatCurve(path, typeof(SpriteRenderer), "m_Enabled");
            var enabledKeyframes = new List<Keyframe>();

            var disabledPrevFrame = false;
            var startTime = GetTimeFromFrame(frames, tag.fromFrame);
            for (var frameIndex = tag.fromFrame; frameIndex < tag.toFrame; ++frameIndex)
            {
                var time = GetTimeFromFrame(frames, frameIndex);
                time -= startTime;

                if (!activeFrames.Contains(frameIndex) && !disabledPrevFrame)
                {
                    var keyframe = GetBoolKeyFrame(false, time);
                    enabledKeyframes.Add(keyframe);
                    disabledPrevFrame = true;
                }
                else if (activeFrames.Contains(frameIndex) && disabledPrevFrame)
                {
                    var keyframe = GetBoolKeyFrame(true, time);
                    enabledKeyframes.Add(keyframe);
                    disabledPrevFrame = false;
                }
            }

            if (enabledKeyframes.Count == 0 && !doesLayerDisableRenderer)
                return;

            // Make sure there is an enable keyframe on the first frame, if the first frame is active.
            if (activeFrames.Contains(tag.fromFrame))
            {
                var keyframe = GetBoolKeyFrame(true, 0f);
                enabledKeyframes.Add(keyframe);
            }

            var animCurve = new AnimationCurve(enabledKeyframes.ToArray());
            AnimationUtility.SetEditorCurve(animationClip, enabledBinding, animCurve);
        }

        static void AddSortOrderKeyframes(Transform layerTransform, Layer layer, Tag tag, IReadOnlyList<Frame> frames, IReadOnlyList<Cell> cells, bool doesLayerHaveCustomSorting, AnimationClip animationClip)
        {
            var layerGo = layerTransform.gameObject;
            var spriteRenderer = layerGo.GetComponent<SpriteRenderer>();
            if (spriteRenderer == null)
                return;

            var sortOrderKeyframes = new List<Keyframe>();
            var path = GetTransformPath(layerTransform);
            var sortOrderBinding = EditorCurveBinding.FloatCurve(path, typeof(SpriteRenderer), "m_SortingOrder");

            var startTime = GetTimeFromFrame(frames, tag.fromFrame);
            var hasKeyOnFirstFrame = false;
            for (var i = 0; i < cells.Count; ++i)
            {
                var previousCell = i > 0 ? cells[i - 1] : default;
                var cell = cells[i];
                if (cell.frameIndex < tag.fromFrame || cell.frameIndex >= tag.toFrame)
                    continue;

                var additiveSortOrder = cell.additiveSortOrder;
                
                // We want to add a keyframe if the current cell has additive sorting, or if the previous cell had additive sorting.
                // This is to ensure that we reset the sort order to 0 if the previous cell had additive sorting.
                if (additiveSortOrder == 0 && previousCell.additiveSortOrder == 0)
                    continue;

                if (cell.frameIndex == tag.fromFrame)
                    hasKeyOnFirstFrame = true;

                var time = GetTimeFromFrame(frames, cell.frameIndex) - startTime;
                var keyframe = GetIntKeyFrame(layer.index + additiveSortOrder, time);
                sortOrderKeyframes.Add(keyframe);
            }

            if (sortOrderKeyframes.Count == 0 && !doesLayerHaveCustomSorting)
                return;

            if (!hasKeyOnFirstFrame)
            {
                var firstFrame = GetIntKeyFrame(layer.index, 0f);
                sortOrderKeyframes.Add(firstFrame);
            }

            var animCurve = new AnimationCurve(sortOrderKeyframes.ToArray());
            AnimationUtility.SetEditorCurve(animationClip, sortOrderBinding, animCurve);
        }

        static float GetTimeFromFrame(IReadOnlyList<Frame> frames, int frameIndex)
        {
            var totalMs = 0;
            for (var i = 0; i < frameIndex; ++i)
                totalMs += frames[i].duration;
            return MsToSeconds(totalMs);
        }

        static float MsToSeconds(int ms) => ms / 1000f;

        static Keyframe GetBoolKeyFrame(bool value, float time)
        {
            var keyframe = new Keyframe();
            keyframe.value = value ? 1f : 0f;
            keyframe.time = time;
            keyframe.inTangent = float.PositiveInfinity;
            keyframe.outTangent = float.PositiveInfinity;
            return keyframe;
        }

        static Keyframe GetIntKeyFrame(int value, float time)
        {
            var keyframe = new Keyframe();
            keyframe.value = value;
            keyframe.time = time;
            keyframe.inTangent = float.PositiveInfinity;
            keyframe.outTangent = float.PositiveInfinity;
            return keyframe;
        }

        static void AddAnimationEvents(Tag tag, IReadOnlyList<Frame> frames, AnimationClip animationClip, bool generateIndividualEvents)
        {
            var events = new List<AnimationEvent>();

            var startTime = GetTimeFromFrame(frames, tag.fromFrame);
            for (var frameIndex = tag.fromFrame; frameIndex < tag.toFrame; ++frameIndex)
            {
                var frame = frames[frameIndex];
                if (frame.eventData.Length == 0)
                    continue;

                var frameTime = GetTimeFromFrame(frames, frameIndex);
                var eventData = frame.eventData;
                for (var m = 0; m < eventData.Length; ++m)
                {
                    if (generateIndividualEvents)
                    {
                        var functionName = eventData[m].Item1;
                        switch (eventData[m].Item2)
                        {
                            case int intParameter:
                                events.Add(new AnimationEvent()
                                {
                                    time = frameTime - startTime,
                                    functionName = functionName,
                                    intParameter = intParameter
                                });
                                break;
                            case float floatParameter:
                                events.Add(new AnimationEvent()
                                {
                                    time = frameTime - startTime,
                                    functionName = functionName,
                                    floatParameter = floatParameter
                                });
                                break;
                            case string stringParameter:
                                events.Add(new AnimationEvent()
                                {
                                    time = frameTime - startTime,
                                    functionName = functionName,
                                    stringParameter = stringParameter,
                                });
                                break;
                            default:
                                events.Add(new AnimationEvent()
                                {
                                    time = frameTime - startTime,
                                    functionName = functionName,
                                });
                                break;
                        }
                    }
                    else
                    {
                        events.Add(new AnimationEvent()
                        {
                            time = frameTime - startTime,
                            functionName = k_CombinedEventName,
                            stringParameter = eventData[m].Item1,
                        });
                    }
                }
            }

            if (events.Count > 0)
                AnimationUtility.SetAnimationEvents(animationClip, events.ToArray());
        }
    }
}
