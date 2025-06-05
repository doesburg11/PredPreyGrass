using System.IO;
using System.Collections.Generic;

using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

using Codice.Client.BaseCommands;
using Codice.Client.Common;
using Unity.PlasticSCM.Editor.AssetUtils.Processor;

namespace Unity.PlasticSCM.Editor.AssetUtils
{
    internal interface ISaveAssets
    {
        void UnderWorkspaceWithConfirmation(
            string wkPath,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            out bool isCancelled);

        void ForChangesWithConfirmation(
            string wkPath,
            List<ChangeInfo> changes,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            out bool isCancelled);

        void ForPathsWithConfirmation(
            string wkPath,
            List<string> paths,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            out bool isCancelled);

        void ForChangesWithoutConfirmation(
            string wkPath,
            List<ChangeInfo> changes,
            WorkspaceOperationsMonitor workspaceOperationsMonitor);

        void ForPathsWithoutConfirmation(
            string wkPath,
            List<string> paths,
            WorkspaceOperationsMonitor workspaceOperationsMonitor);
    }

    internal class SaveAssets : ISaveAssets
    {
        void ISaveAssets.UnderWorkspaceWithConfirmation(
            string wkPath,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            out bool isCancelled)
        {
            ForPaths(
                wkPath,
                null,
                true,
                workspaceOperationsMonitor,
                out isCancelled);
        }

        void ISaveAssets.ForChangesWithConfirmation(
            string wkPath,
            List<ChangeInfo> changes,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            out bool isCancelled)
        {
            ForPaths(
                wkPath,
                GetPaths(changes),
                true,
                workspaceOperationsMonitor,
                out isCancelled);
        }

        void ISaveAssets.ForPathsWithConfirmation(
            string wkPath,
            List<string> paths,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            out bool isCancelled)
        {
            ForPaths(
                wkPath,
                paths,
                true,
                workspaceOperationsMonitor,
                out isCancelled);
        }

        void ISaveAssets.ForChangesWithoutConfirmation(
            string wkPath,
            List<ChangeInfo> changes,
            WorkspaceOperationsMonitor workspaceOperationsMonitor)
        {
            bool isCancelled;
            ForPaths(
                wkPath,
                GetPaths(changes),
                false,
                workspaceOperationsMonitor,
                out isCancelled);
        }

        void ISaveAssets.ForPathsWithoutConfirmation(
            string wkPath,
            List<string> paths,
            WorkspaceOperationsMonitor workspaceOperationsMonitor)
        {
            bool isCancelled;
            ForPaths(
                wkPath,
                paths,
                false,
                workspaceOperationsMonitor,
                out isCancelled);
        }

        static void ForPaths(
            string wkPath,
            List<string> paths,
            bool askForUserConfirmation,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            out bool isCancelled)
        {
            workspaceOperationsMonitor.Disable();
            try
            {
                SaveDirtyScenes(
                    wkPath,
                    paths,
                    askForUserConfirmation,
                    out isCancelled);

                if (isCancelled)
                    return;

                AssetDatabase.SaveAssets();
            }
            finally
            {
                workspaceOperationsMonitor.Enable();
            }
        }

        static void SaveDirtyScenes(
            string wkPath,
            List<string> paths,
            bool askForUserConfirmation,
            out bool isCancelled)
        {
            isCancelled = false;

            List<Scene> scenesToSave = GetScenesToSave(wkPath, paths);

            if (scenesToSave.Count == 0)
                return;

            if (askForUserConfirmation)
            {
                isCancelled = !EditorSceneManager.
                    SaveModifiedScenesIfUserWantsTo(
                        scenesToSave.ToArray());

                if (!isCancelled)
                    DiscardChangesInActiveSceneIfDirty(scenesToSave);

                return;
            }

            EditorSceneManager.SaveScenes(
                scenesToSave.ToArray());
        }

        static void DiscardChangesInActiveSceneIfDirty(List<Scene> scenesToSave)
        {
            string activeScenePath = EditorSceneManager.GetActiveScene().path;
            Scene? activeScene = GetSceneByPath(scenesToSave, activeScenePath);

            if (activeScene == null)
                return;

            if (!activeScene.Value.isDirty)
                return;

            EditorSceneManager.OpenScene(activeScenePath);
        }

        static Scene? GetSceneByPath(List<Scene> scenes, string scenePath)
        {
            foreach (Scene scene in scenes)
            {
                if (scene.path == scenePath)
                    return scene;
            }

            return null;
        }

        static List<Scene> GetScenesToSave(string wkPath, List<string> paths)
        {
            List<Scene> dirtyScenes = GetDirtyScenesUnderWorkspace(wkPath);

            if (paths == null)
                return dirtyScenes;

            List<Scene> scenesToSave = new List<Scene>();

            foreach (Scene dirtyScene in dirtyScenes)
            {
                if (Contains(paths, dirtyScene))
                    scenesToSave.Add(dirtyScene);
            }

            return scenesToSave;
        }

        static List<Scene> GetDirtyScenesUnderWorkspace(string wkPath)
        {
            List<Scene> dirtyScenes = new List<Scene>();

            for (int i = 0; i < SceneManager.sceneCount; i++)
            {
                Scene scene = SceneManager.GetSceneAt(i);

                if (!scene.isDirty)
                    continue;

                if (string.IsNullOrEmpty(scene.path))
                    continue;

                string fullPath = Path.GetFullPath(scene.path);

                if (!PathHelper.IsContainedOn(fullPath, wkPath))
                    continue;

                dirtyScenes.Add(scene);
            }

            return dirtyScenes;
        }

        static bool Contains(
            List<string> paths,
            Scene scene)
        {
            foreach (string path in paths)
            {
                if (PathHelper.IsSamePath(
                        path,
                        Path.GetFullPath(scene.path)))
                    return true;
            }

            return false;
        }

        static List<string> GetPaths(
            List<ChangeInfo> changeInfos)
        {
            List<string> result = new List<string>();
            foreach (ChangeInfo change in changeInfos)
                result.Add(change.GetFullPath());
            return result;
        }
    }
}
