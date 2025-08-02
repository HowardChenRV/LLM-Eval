pipeline {
    agent none  // Global agent not specified, each stage specifies separately
    
    environment {
        // Harbor configuration
        HARBOR_REGISTRY = 'harbor.xxx.com'
        HARBOR_PROJECT = 'xxx'
        IMAGE_NAME = 'llm-eval'
        FULL_IMAGE_NAME = "${HARBOR_REGISTRY}/${HARBOR_PROJECT}/${IMAGE_NAME}"
        
        // Windows shared directory configuration
        SHARED_DIR = '\\\\LAPTOP-RJ6A3U9I\\aipc_public\\LLM-Eval'
    }
    
    stages {
        stage('Extract Tag') {
            agent any
            steps {
                script {
                    if (env.gitlabBranch?.startsWith('refs/tags/')) {
                        env.EXTRACTED_TAG = env.gitlabBranch.replace('refs/tags/', '').replace('/', '_')
                        echo "Extracted Tag: ${env.EXTRACTED_TAG}"
                    } else {
                        error("Current gitlabBranch is not in Tag format: ${env.gitlabBranch}")
                    }
                }
            }
        }
        
        stage('Parallel Build Images') {
            parallel {
                stage('Linux Build Process') {
                    agent { label 'is-dbgnpqtk6l5i674c' }
                    stages {
                        stage('Linux: Code Checkout') {
                            steps {
                                cleanWs()
                                checkout([
                                    $class: 'GitSCM',
                                    branches: [[name: env.gitlabBranch]],
                                    extensions: [
                                        [$class: 'CloneOption',
                                         depth: 1,
                                         noTags: false,
                                         shallow: true,
                                         reference: '',
                                         timeout: 10],
                                        [$class: 'CleanBeforeCheckout']
                                    ],
                                    userRemoteConfigs: [[
                                        url: scm.userRemoteConfigs[0].url,
                                        credentialsId: scm.userRemoteConfigs[0].credentialsId
                                    ]]
                                ])
                            }
                        }
                        stage('Linux: Build Image') {
                            steps {
                                script {
                                    withCredentials([usernamePassword(
                                        credentialsId: 'harbor-creds',
                                        usernameVariable: 'HARBOR_USER',
                                        passwordVariable: 'HARBOR_PASSWORD'
                                    )]) {
                                        sh """
                                            echo \$HARBOR_PASSWORD | docker login -u \$HARBOR_USER --password-stdin ${HARBOR_REGISTRY}
                                        """
                                        
                                        def imageTag = env.EXTRACTED_TAG ?: 'latest'
                                        docker.build("${FULL_IMAGE_NAME}:${imageTag}", "--no-cache --pull --file docker/jenkins_serving/Dockerfile .")
                                        docker.image("${FULL_IMAGE_NAME}:${imageTag}").push()
                                        
                                        echo "✅ Linux image pushed successfully"
                                        echo "🔗 Pull command: docker pull ${FULL_IMAGE_NAME}:${imageTag}"
                                    }
                                }
                            }
                            post {
                                always {
                                    sh "docker logout ${HARBOR_REGISTRY}"
                                    cleanWs()   // Clean workspace
                                    echo "✅ Linux build completed, 🧹 workspace cleaned"
                                }
                            }
                        }
                    }
                }
                
                stage('Windows Build Process') {
                    agent { label 'chip-perf' }
                    stages {
                        stage('Windows: Code Checkout') {
                            steps { checkout scm }
                        }
                        
                        stage('Windows: Package exe') {
                            steps {
                                script {
                                    // 1. Install dependencies
                                    powershell """
                                        python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
                                        pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
                                        pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
                                    """
                                    
                                    // 2. Build EXE
                                    powershell "python build_windows_exe.py"
                                    
                                    // 3. Create target directory and copy EXE
                                    powershell """
                                        \$targetDir = \"${env.SHARED_DIR}\\${env.EXTRACTED_TAG}\"
                                        New-Item -Path \$targetDir -ItemType Directory -Force
                                        Copy-Item -Path \"dist\\*.exe\" -Destination \$targetDir -Force
                                        Write-Host \"✅ EXE copied to: \$targetDir\"
                                    """
                                    
                                    // 4. Archive artifacts to Jenkins
                                    archiveArtifacts artifacts: 'dist/**/*.exe', fingerprint: true
                                }
                            }
                            post {
                                always {
                                    // Clean temporary files
                                    bat 'rd /s /q build dist || true'
                                    cleanWs()   // Clean workspace
                                    echo "✅ Windows build completed, 🧹 workspace cleaned"
                                }
                                success {
                                    echo "📦 Artifact archive path: ${BUILD_URL}artifact/"
                                    echo "📂 Shared directory path: ${env.SHARED_DIR}\\${env.EXTRACTED_TAG}"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    post {
        success {
            script {
                echo "🎉 All build tasks completed!"
                echo "🐳 Image: ${FULL_IMAGE_NAME}:${env.EXTRACTED_TAG}"
                // echo "🖥️  EXE location: ${env.SHARED_DIR}\\${env.EXTRACTED_TAG}"
            }
        }
    }
}
