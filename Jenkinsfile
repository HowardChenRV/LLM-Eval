pipeline {
    agent none  // 全局不指定 agent，各阶段单独指定

    environment {
        // Harbor 配置
        HARBOR_REGISTRY = 'harbor.xxx.com'
        HARBOR_PROJECT = 'xxx'
        IMAGE_NAME = 'llm-eval'
        FULL_IMAGE_NAME = "${HARBOR_REGISTRY}/${HARBOR_PROJECT}/${IMAGE_NAME}"
        
        // Windows 共享目录配置
        SHARED_DIR = '\\\\LAPTOP-RJ6A3U9I\\aipc_public\\LLM-Eval'
    }

    stages {
        stage('提取 Tag') {
            agent any
            steps {
                script {
                    if (env.gitlabBranch?.startsWith('refs/tags/')) {
                        env.EXTRACTED_TAG = env.gitlabBranch.replace('refs/tags/', '').replace('/', '_')
                        echo "提取到的 Tag: ${env.EXTRACTED_TAG}"
                    } else {
                        error("当前 gitlabBranch 不是 Tag 格式: ${env.gitlabBranch}")
                    }
                }
            }
        }

        stage('并行构建镜像') {
            parallel {
                stage('Linux 构建流程') {
                    agent { label 'is-dbgnpqtk6l5i674c' }
                    stages {
                        stage('Linux: 代码检出') {
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
                        stage('Linux: 构建镜像') {
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
                                        
                                        echo "✅ Linux 镜像推送成功"
                                        echo "🔗 拉取命令: docker pull ${FULL_IMAGE_NAME}:${imageTag}"
                                    }
                                }
                            }
                            post {
                                always {
                                    sh "docker logout ${HARBOR_REGISTRY}"
                                    cleanWs()   // 清理工作区
                                    echo "✅ Linux 构建完成，🧹 工作区已清理"
                                }
                            }
                        }
                    }
                }

                stage('Windows 构建流程') {
                    agent { label 'chip-perf' }
                    stages {
                        stage('Windows: 代码检出') {
                            steps { checkout scm }
                        }
                        
                        stage('Windows: 打包exe') {
                            steps {
                                script {
                                    // 1. 安装依赖
                                    powershell """
                                        python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
                                        pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
                                        pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
                                    """
                                    
                                    // 2. 构建EXE
                                    powershell "python build_windows_exe.py"
                                    
                                    // 3. 创建目标目录并复制EXE
                                    powershell """
                                        \$targetDir = \"${env.SHARED_DIR}\\${env.EXTRACTED_TAG}\"
                                        New-Item -Path \$targetDir -ItemType Directory -Force
                                        Copy-Item -Path \"dist\\*.exe\" -Destination \$targetDir -Force
                                        Write-Host \"✅ EXE已复制到: \$targetDir\"
                                    """
                                    
                                    // 4. 存档产物到Jenkins
                                    archiveArtifacts artifacts: 'dist/**/*.exe', fingerprint: true
                                }
                            }
                            post {
                                always {
                                    // 清理临时文件
                                    bat 'rd /s /q build dist || true'
                                    cleanWs()   // 清理工作区
                                    echo "✅ Windows 构建完成，🧹 工作区已清理"
                                }
                                success {
                                    echo "📦 产物存档路径: ${BUILD_URL}artifact/"
                                    echo "📂 共享目录路径: ${env.SHARED_DIR}\\${env.EXTRACTED_TAG}"
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
                echo "🎉 所有构建任务完成！"
                echo "🐳 镜像: ${FULL_IMAGE_NAME}:${env.EXTRACTED_TAG}"
                // echo "🖥️  EXE位置: ${env.SHARED_DIR}\\${env.EXTRACTED_TAG}"
            }
        }
    }
}
