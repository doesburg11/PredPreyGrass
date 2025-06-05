using System.IO;
using MyExercise_6s;
using NUnit.Framework;

namespace Tests_6s
{
    internal class FileCreatorTests
    {
        [SetUp]
        public void Setup()
        {
            Directory.CreateDirectory(FileCreator.k_Directory);
        }
        
        [Test]
        public void CreatesEmptyFile()
        {
            var fileCreatorUnderTest = new FileCreator();
            var expectedFileName = "MyEmptyFile.txt";
            
            fileCreatorUnderTest.CreateEmptyFile(expectedFileName);

            var files = Directory.GetFiles(FileCreator.k_Directory);
            Assert.That(files.Length, Is.EqualTo(1), "Expected one file.");
            var expectedFilePath = Path.Combine(FileCreator.k_Directory, expectedFileName);
            Assert.That(files[0], Is.EqualTo(expectedFilePath));
        }
        
        [Test]
        public void CreatesFile()
        {
            var fileCreatorUnderTest = new FileCreator();
            var expectedFileName = "MyFile.txt";
            var expectedContent = "TheFileContent";
            
            fileCreatorUnderTest.CreateFile(expectedFileName, expectedContent);

            var files = Directory.GetFiles(FileCreator.k_Directory);
            Assert.That(files.Length, Is.EqualTo(1), "Expected one file.");
            var expectedFilePath = Path.Combine(FileCreator.k_Directory, expectedFileName);
            Assert.That(files[0], Is.EqualTo(expectedFilePath));
            var content = File.ReadAllText(expectedFilePath);
            Assert.That(content, Is.EqualTo(expectedContent));
        }
        
        [TearDown]
        public void Teardown()
        {
            Directory.Delete(FileCreator.k_Directory, true);
        }
    }
}