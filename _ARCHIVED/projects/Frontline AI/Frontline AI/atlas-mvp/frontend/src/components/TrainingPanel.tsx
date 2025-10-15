import React, { useState, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useDropzone } from 'react-dropzone';
import { trainingAPI } from '../services/api';

interface Model {
  id: number;
  name: string;
  path: string;
  created_at: string;
  accuracy: number;
  is_active: boolean;
}

interface Project {
  name: string;
  path: string;
  image_count: number;
  has_annotations: boolean;
  created_at: string;
}

const TrainingPanel: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [projectName, setProjectName] = useState('');
  const [trainingEpochs, setTrainingEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(16);
  const [activeTab, setActiveTab] = useState<'upload' | 'projects' | 'models'>('upload');
  const queryClient = useQueryClient();

  // Query for models
  const { data: modelsData } = useQuery({
    queryKey: ['models'],
    queryFn: () => trainingAPI.getModels(),
    select: (response) => response.data.models as Model[]
  });

  // Query for projects  
  const { data: projectsData } = useQuery({
    queryKey: ['projects'],
    queryFn: () => trainingAPI.getProjects(),
    select: (response) => response.data.projects as Project[]
  });

  // Mutation for uploading images
  const uploadImagesMutation = useMutation({
    mutationFn: ({ files, projectName }: { files: File[], projectName: string }) => {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      formData.append('project_name', projectName);
      return trainingAPI.uploadImages(formData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      setSelectedFiles([]);
      setProjectName('');
    }
  });

  // Mutation for starting training
  const startTrainingMutation = useMutation({
    mutationFn: ({ projectName, epochs, batchSize }: { projectName: string, epochs: number, batchSize: number }) =>
      trainingAPI.startTraining(projectName, epochs, batchSize),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    }
  });

  // Mutation for activating model
  const activateModelMutation = useMutation({
    mutationFn: (modelId: number) => trainingAPI.activateModel(modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    }
  });

  // Mutation for deleting project
  const deleteProjectMutation = useMutation({
    mutationFn: (projectName: string) => trainingAPI.deleteProject(projectName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    }
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setSelectedFiles(prev => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: true
  });

  const handleUpload = () => {
    if (selectedFiles.length > 0 && projectName.trim()) {
      uploadImagesMutation.mutate({ files: selectedFiles, projectName: projectName.trim() });
    }
  };

  const handleStartTraining = (project: Project) => {
    startTrainingMutation.mutate({ 
      projectName: project.name, 
      epochs: trainingEpochs, 
      batchSize: batchSize 
    });
  };

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const models = modelsData || [];
  const projects = projectsData || [];

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-white">AutoML Training</h2>
          <div className="flex space-x-1 bg-gray-700 rounded-lg p-1">
            {[
              { id: 'upload', label: 'Upload Data', icon: '📁' },
              { id: 'projects', label: 'Projects', icon: '🗂️' },
              { id: 'models', label: 'Models', icon: '🧠' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:text-white hover:bg-gray-600'
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="space-y-6">
            <div>
              <label className="block text-white font-medium mb-2">Project Name</label>
              <input
                type="text"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                placeholder="Enter project name"
                className="w-full px-4 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
                isDragActive
                  ? 'border-blue-500 bg-blue-900 bg-opacity-20'
                  : 'border-gray-600 hover:border-gray-500 hover:bg-gray-700'
              }`}
            >
              <input {...getInputProps()} />
              <div className="text-4xl mb-4">📷</div>
              <p className="text-white font-medium mb-2">
                Drop training images here, or click to select files
              </p>
              <p className="text-gray-400 text-sm">
                Supported formats: JPG, PNG. Upload 20-100 images for best results.
              </p>
            </div>

            {selectedFiles.length > 0 && (
              <div>
                <h3 className="text-white font-medium mb-3">Selected Files ({selectedFiles.length})</h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {selectedFiles.map((file, index) => (
                    <div key={index} className="relative bg-gray-700 rounded-lg p-2">
                      <img
                        src={URL.createObjectURL(file)}
                        alt={file.name}
                        className="w-full h-20 object-cover rounded mb-2"
                      />
                      <p className="text-white text-xs truncate">{file.name}</p>
                      <button
                        onClick={() => removeFile(index)}
                        className="absolute -top-2 -right-2 w-6 h-6 bg-red-600 text-white rounded-full text-xs hover:bg-red-700"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={selectedFiles.length === 0 || !projectName.trim() || uploadImagesMutation.isPending}
              className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors"
            >
              {uploadImagesMutation.isPending ? 'Uploading...' : 'Upload Images'}
            </button>

            {uploadImagesMutation.isError && (
              <div className="p-3 bg-red-900 bg-opacity-50 border border-red-700 rounded-lg">
                <p className="text-red-300">Upload failed. Please try again.</p>
              </div>
            )}

            {uploadImagesMutation.isSuccess && (
              <div className="p-3 bg-green-900 bg-opacity-50 border border-green-700 rounded-lg">
                <p className="text-green-300">Images uploaded successfully!</p>
              </div>
            )}
          </div>
        )}

        {/* Projects Tab */}
        {activeTab === 'projects' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-white">Training Projects</h3>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <label className="text-white text-sm">Epochs:</label>
                  <select
                    value={trainingEpochs}
                    onChange={(e) => setTrainingEpochs(Number(e.target.value))}
                    className="bg-gray-700 text-white rounded px-2 py-1 text-sm"
                  >
                    <option value={5}>5</option>
                    <option value={10}>10</option>
                    <option value={20}>20</option>
                    <option value={50}>50</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  <label className="text-white text-sm">Batch Size:</label>
                  <select
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    className="bg-gray-700 text-white rounded px-2 py-1 text-sm"
                  >
                    <option value={8}>8</option>
                    <option value={16}>16</option>
                    <option value={32}>32</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {projects.map((project) => (
                <div key={project.name} className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <h4 className="text-white font-medium">{project.name}</h4>
                    <button
                      onClick={() => deleteProjectMutation.mutate(project.name)}
                      className="text-red-400 hover:text-red-300 text-sm"
                    >
                      🗑️
                    </button>
                  </div>
                  
                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-300">Images:</span>
                      <span className="text-white">{project.image_count}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-300">Annotations:</span>
                      <span className={project.has_annotations ? 'text-green-400' : 'text-red-400'}>
                        {project.has_annotations ? 'Ready' : 'Missing'}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-300">Created:</span>
                      <span className="text-white">
                        {new Date(project.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <button
                      onClick={() => handleStartTraining(project)}
                      disabled={
                        !project.has_annotations || 
                        project.image_count < 5 || 
                        startTrainingMutation.isPending
                      }
                      className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded font-medium text-sm transition-colors"
                    >
                      {startTrainingMutation.isPending ? 'Training...' : 'Start Training'}
                    </button>
                    
                    {(!project.has_annotations || project.image_count < 5) && (
                      <p className="text-yellow-400 text-xs text-center">
                        {!project.has_annotations ? 'Annotations required' : 'Need at least 5 images'}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {projects.length === 0 && (
              <div className="text-center py-12">
                <div className="text-4xl mb-4">📂</div>
                <p className="text-gray-400">No training projects yet</p>
                <p className="text-gray-500 text-sm">Upload some images to get started</p>
              </div>
            )}

            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-4">
              <h4 className="text-blue-300 font-medium mb-2">Training Instructions</h4>
              <ol className="text-blue-200 text-sm space-y-1">
                <li>1. Upload 20-100 training images per class</li>
                <li>2. Use the annotation tool to label objects in your images</li>
                <li>3. Start training with appropriate epochs (10-20 for most cases)</li>
                <li>4. Training typically takes 5-10 minutes on M1 MacBook</li>
                <li>5. Activate your trained model when ready</li>
              </ol>
            </div>
          </div>
        )}

        {/* Models Tab */}
        {activeTab === 'models' && (
          <div className="space-y-6">
            <h3 className="text-lg font-medium text-white">Trained Models</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {models.map((model) => (
                <div
                  key={model.id}
                  className={`rounded-lg p-4 border-2 ${
                    model.is_active
                      ? 'bg-green-900 bg-opacity-30 border-green-700'
                      : 'bg-gray-700 border-gray-600'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <h4 className="text-white font-medium">{model.name}</h4>
                    {model.is_active && (
                      <span className="px-2 py-1 bg-green-600 text-white text-xs rounded">
                        Active
                      </span>
                    )}
                  </div>
                  
                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-300">Accuracy:</span>
                      <span className={`font-medium ${
                        model.accuracy > 0.9 
                          ? 'text-green-400'
                          : model.accuracy > 0.8
                          ? 'text-yellow-400'
                          : 'text-red-400'
                      }`}>
                        {(model.accuracy * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-300">Created:</span>
                      <span className="text-white">
                        {new Date(model.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  {!model.is_active && (
                    <button
                      onClick={() => activateModelMutation.mutate(model.id)}
                      disabled={activateModelMutation.isPending}
                      className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded font-medium text-sm transition-colors"
                    >
                      {activateModelMutation.isPending ? 'Activating...' : 'Activate Model'}
                    </button>
                  )}
                </div>
              ))}
            </div>

            {models.length === 0 && (
              <div className="text-center py-12">
                <div className="text-4xl mb-4">🧠</div>
                <p className="text-gray-400">No trained models yet</p>
                <p className="text-gray-500 text-sm">Train your first custom model to get started</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingPanel;