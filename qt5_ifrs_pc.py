


import sys
import os
import re
import uuid
import requests
import base64
import fitz  # PyMuPDF

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QTextEdit, QPushButton, QComboBox, 
                             QMessageBox, QFileDialog, QTabWidget, QGroupBox)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

# ==========================================
# [백그라운드 가속 스레드] 대용량 PDF 처리 전용
# ==========================================
class PdfInjectorThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, file_path, standard_tag, text_storage):
        super().__init__()
        self.file_path = file_path
        self.standard_tag = standard_tag
        self.text_storage = text_storage

    def run(self):
        try:
            self.progress_signal.emit("⏳ 로컬 PDF 파일 분석 및 텍스트 추출 시작...")
            doc = fitz.open(self.file_path)
            full_text = ""
            for page_num, page in enumerate(doc):
                full_text += page.get_text() + "\n"
                if page_num % 50 == 0 and page_num > 0:
                    self.progress_signal.emit(f" 진행 중: {page_num} 페이지 추출 완료...")

            self.progress_signal.emit("✂️ K-IFRS 조항 패턴 기반 정밀 청킹 분할 중...")
            article_pattern = r"(제\s*\d+\s*조\s*[^제]+)"
            chunks = re.findall(article_pattern, full_text)
            if not chunks:
                chunks = [full_text[i:i+800] for i in range(0, len(full_text), 600)]
                
            total_chunks = len(chunks)
            self.progress_signal.emit(f"📦 총 {total_chunks}개의 회계 청크 감지. 로컬 메모리 주입 시작...")

            injected_count = 0
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < 20: continue
                article_match = re.search(r"제\s*(\d+)\s*조", chunk)
                art_no = article_match.group(1) if article_match else f"p_{injected_count}"
                
                structured_text = f"[{self.standard_tag} 기준서 제{art_no}조]\n{chunk}"
                self.text_storage.append({"text": structured_text, "tag": self.standard_tag})
                injected_count += 1
                
            self.finished_signal.emit(injected_count)
        except Exception as e:
            self.error_signal.emit(str(e))


# ==========================================
# [범용 Ollama 백그라운드 워커] – RAG와 Vision 모두 사용
# ==========================================
class OllamaWorker(QObject):
    finished = pyqtSignal(str)      # 성공 시 답변 텍스트
    error = pyqtSignal(str)         # 오류 메시지
    progress = pyqtSignal(str)      # 진행 상황 (선택적)

    def __init__(self, url, payload):
        super().__init__()
        self.url = url
        self.payload = payload

    def run(self):
        try:
            self.progress.emit("⏳ Ollama 서버에 연결 중...")
            response = requests.post(self.url, json=self.payload, timeout=120)
            if response.status_code == 200:
                result = response.json().get('response', '응답 필드 누락')
                self.finished.emit(result)
            else:
                self.error.emit(f"HTTP {response.status_code}: {response.text[:200]}")
        except Exception as e:
            self.error.emit(str(e))


# ==========================================
# [메인 마스터 터미널 윈도우]
# ==========================================
class IfrsDualUltimateTerminal(QWidget):
    def __init__(self):
        super().__init__()
        # 올바른 Ollama API 엔드포인트 (전체 주소)
        self.OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
        self.selected_file_path = ""
        self.selected_image_path = ""
        self.local_text_storage = [] 

        # 백그라운드 스레드/워커 참조 저장
        self.rag_thread = None
        self.rag_worker = None
        self.vision_thread = None
        self.vision_worker = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('🏢 K-IFRS 듀얼 지능형 AI 분석 플랫폼 V1.5 (Local Master)')
        self.resize(900, 750)
        self.setStyleSheet("background-color: #1e272e; color: #dcdde1;")

        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { background: #2f3640; padding: 12px; font-weight: bold; color: #dcdde1; } "
                                "QTabBar::tab:selected { background: #192a56; color: #00d2d3; }")
        
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab1, "📝 텍스트 조항 RAG 엔진")
        self.tabs.addTab(self.tab2, "🖼️ 도표/그래프 비전 엔진")
        
        self.setup_tab1_text_rag()
        self.setup_tab2_vision_multimodal()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def setup_tab1_text_rag(self):
        layout = QVBoxLayout()
        group_inject = QGroupBox("📂 고부가가치 데이터셋 주입 제어기")
        group_inject.setStyleSheet("color: #00d2d3; font-weight: bold; border: 1px solid #718093; padding: 10px;")
        g_layout = QHBoxLayout()
        
        self.lbl_file = QLabel('선택된 IFRS PDF 파일 없음', self)
        self.lbl_file.setStyleSheet("color: #dcdde1;")
        self.btn_browse = QPushButton('파일 선택')
        self.btn_browse.setStyleSheet("background-color: #3f51b5; color: white; padding: 6px;")
        self.btn_browse.clicked.connect(self.select_pdf_file)
        
        self.btn_inject = QPushButton('금고에 주입')
        self.btn_inject.setStyleSheet("background-color: #e67e22; color: white; padding: 6px;")
        self.btn_inject.clicked.connect(self.start_injection_thread)
        self.btn_inject.setEnabled(False)
        
        g_layout.addWidget(self.lbl_file, 4)
        g_layout.addWidget(self.btn_browse, 1)
        g_layout.addWidget(self.btn_inject, 1)
        group_inject.setLayout(g_layout)
        layout.addWidget(group_inject)

        input_layout = QHBoxLayout()
        self.standard_combo = QComboBox()
        self.standard_combo.addItems(['IFRS17', 'IFRS9', 'IFRS15', '전체조회'])
        self.standard_combo.setStyleSheet("padding: 8px; background-color: white; color: black; font-weight: bold;")
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText('로컬 AI에게 질문할 회계 기준서 내용을 입력하세요...')
        self.query_input.setStyleSheet("padding: 8px; background-color: white; color: black;")
        self.query_input.returnPressed.connect(self.execute_rag_pipeline)
        
        input_layout.addWidget(self.standard_combo, 1)
        input_layout.addWidget(self.query_input, 4)
        layout.addLayout(input_layout)

        self.btn_run = QPushButton('⚡ 로컬 하드웨어 가속 RAG 분석 실행')
        self.btn_run.setFont(QFont('Arial', 10, QFont.Bold))
        self.btn_run.setStyleSheet("background-color: #4cd137; color: white; padding: 12px;")
        self.btn_run.clicked.connect(self.execute_rag_pipeline)
        layout.addWidget(self.btn_run)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet("background-color: #2f3640; color: #00d2d3; font-family: Consolas; font-size: 11pt; padding: 12px;")
        self.result_display.setText("✓ 텍스트 RAG 시스템 대기 중...")
        layout.addWidget(self.result_display)
        self.tab1.setLayout(layout)

    def setup_tab2_vision_multimodal(self):
        layout = QVBoxLayout()
        vision_layout = QHBoxLayout()
        self.lbl_image = QLabel('🖼️ 선택된 회계 도표/그래프 이미지 없음', self)
        self.lbl_image.setStyleSheet("color: #dcdde1; font-weight: bold;")
        
        self.btn_img_browse = QPushButton('이미지 선택')
        self.btn_img_browse.setStyleSheet("background-color: #9c27b0; color: white; padding: 6px;")
        self.btn_img_browse.clicked.connect(self.select_image_file)
        
        self.btn_vision_run = QPushButton('🔍 표/그래프 독해 실행')
        self.btn_vision_run.setStyleSheet("background-color: #0097e6; color: white; padding: 6px;")
        self.btn_vision_run.clicked.connect(self.execute_vision_pipeline)
        self.btn_vision_run.setEnabled(False)
        
        vision_layout.addWidget(self.lbl_image, 4)
        vision_layout.addWidget(self.btn_img_browse, 1)
        vision_layout.addWidget(self.btn_vision_run, 1)
        layout.addLayout(vision_layout)

        self.img_preview = QLabel("선택된 이미지 미리보기 구역")
        self.img_preview.setAlignment(Qt.AlignCenter)
        self.img_preview.setStyleSheet("border: 1px dashed #718093; background-color: #2f3640;")
        self.img_preview.setFixedHeight(120)
        layout.addWidget(self.img_preview)

        v_input_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(['minicpm-v', 'qwen2.5'])
        self.model_combo.setStyleSheet("padding: 8px; background-color: white; color: black; font-weight: bold;")
        
        self.vision_query_input = QLineEdit()
        self.vision_query_input.setPlaceholderText('도표 이미지에 대해 분석할 구체적인 질문을 입력하세요...')
        self.vision_query_input.setStyleSheet("padding: 8px; background-color: white; color: black;")
        self.vision_query_input.returnPressed.connect(self.execute_vision_pipeline)
        
        v_input_layout.addWidget(self.model_combo, 1)
        v_input_layout.addWidget(self.vision_query_input, 4)
        layout.addLayout(v_input_layout)

        self.vision_result_display = QTextEdit()
        self.vision_result_display.setReadOnly(True)
        self.vision_result_display.setStyleSheet("background-color: #2f3640; color: #ff9f43; font-family: Consolas; font-size: 11pt; padding: 12px;")
        self.vision_result_display.setText("✓ 멀티모달 비전 시스템 대기 중...")
        layout.addWidget(self.vision_result_display)
        self.tab2.setLayout(layout)

    # ---------- PDF 주입 (기존 코드 그대로) ----------
    def select_pdf_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'IFRS 원본 PDF 선택', '', 'PDF Files (*.pdf)')
        if file_name:
            self.selected_file_path = file_name
            self.lbl_file.setText(f"📂 로드됨: {os.path.basename(file_name)}")
            self.btn_inject.setEnabled(True)

    def start_injection_thread(self):
        if not self.selected_file_path: return
        selected_std = self.standard_combo.currentText()
        std_tag = "IFRS" if selected_std == "전체조회" else selected_std
        self.btn_inject.setEnabled(False)
        self.btn_browse.setEnabled(False)
        
        self.injector_thread = PdfInjectorThread(self.selected_file_path, std_tag, self.local_text_storage)
        self.injector_thread.progress_signal.connect(self.update_terminal_log)
        self.injector_thread.finished_signal.connect(self.injection_finished)
        self.injector_thread.error_signal.connect(self.injection_error)
        self.injector_thread.start()

    def update_terminal_log(self, msg):
        self.result_display.append(msg)
        self.result_display.ensureCursorVisible()

    def injection_finished(self, count):
        self.result_display.append(f"\n🎯 [성공] 총 {count}개의 데이터 청크가 엔진 메모리에 주입되었습니다.")
        self.btn_browse.setEnabled(True)
        self.btn_inject.setEnabled(True)

    def injection_error(self, error_msg):
        QMessageBox.critical(self, '주입 오류', f'예외 발생: {error_msg}')
        self.btn_browse.setEnabled(True)
        self.btn_inject.setEnabled(True)

    # ---------- RAG 파이프라인 (백그라운드 스레드 적용) ----------
    def execute_rag_pipeline(self):
        question = self.query_input.text().strip()
        if not question:
            return
        selected_std = self.standard_combo.currentText()

        # UI 비활성화
        self.btn_run.setEnabled(False)
        self.result_display.clear()
        self.result_display.append("🔍 1단계: 로컬 메모리 금고에서 고속 키워드 컨텍스트 매칭 중...")

        # 1. 로컬 매칭 (빠르므로 메인 스레드에서 처리)
        matched_chunks = []
        for item in self.local_text_storage:
            if selected_std != "전체조회" and item["tag"] != selected_std:
                continue
            if any(word in item["text"] for word in question.split() if len(word) >= 2):
                matched_chunks.append(item["text"])
                if len(matched_chunks) >= 2:
                    break

        retrieved_context = "\n\n".join(matched_chunks) if matched_chunks else "로컬 메모리 내에 관련 조항이 없습니다."

        self.result_display.append("🧠 2단계: 근거 확보 완료. Ollama로 전송 중 (백그라운드)...")

        # 2. 프롬프트 구성
        prompt = f"[IFRS 근거]:\n{retrieved_context}\n\n[질의]: {question}\n\n위 근거에만 철저히 기반하여 한국어로 정밀 답변하세요."
        payload = {"model": "qwen2.5", "prompt": prompt, "stream": False}

        # 3. 백그라운드 스레드 시작
        self.rag_thread = QThread()
        self.rag_worker = OllamaWorker(self.OLLAMA_URL, payload)
        self.rag_worker.moveToThread(self.rag_thread)

        # 시그널 연결
        self.rag_worker.progress.connect(self.update_rag_progress)
        self.rag_worker.finished.connect(self.on_rag_finished)
        self.rag_worker.error.connect(self.on_rag_error)

        self.rag_thread.started.connect(self.rag_worker.run)
        self.rag_thread.finished.connect(self.rag_thread.deleteLater)  # 자동 정리
        self.rag_thread.start()

    def update_rag_progress(self, msg):
        self.result_display.append(msg)
        self.result_display.ensureCursorVisible()

    def on_rag_finished(self, answer):
        self.result_display.append(f"\n🏆 [로컬 RAG 분석 성공]\n\n{answer}")
        self.btn_run.setEnabled(True)
        # 스레드 정리
        if self.rag_thread and self.rag_thread.isRunning():
            self.rag_thread.quit()
            self.rag_thread.wait()
        self.rag_thread = None
        self.rag_worker = None

    def on_rag_error(self, error_msg):
        self.result_display.append(f"\n❌ 오류: {error_msg}")
        self.btn_run.setEnabled(True)
        if self.rag_thread and self.rag_thread.isRunning():
            self.rag_thread.quit()
            self.rag_thread.wait()
        self.rag_thread = None
        self.rag_worker = None

    # ---------- Vision 파이프라인 (백그라운드 스레드 적용) ----------
    def select_image_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '회계 도표 이미지 선택', '', 'Image Files (*.png *.jpg *.jpeg)')
        if file_name:
            self.selected_image_path = file_name
            self.lbl_image.setText(f"🖼️ 이미지 로드됨: {os.path.basename(file_name)}")
            self.btn_vision_run.setEnabled(True)
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(self.img_preview.width(), self.img_preview.height(), Qt.KeepAspectRatio)
            self.img_preview.setPixmap(scaled_pixmap)

    def execute_vision_pipeline(self):
        question = self.vision_query_input.text().strip()
        if not question:
            question = "이 표의 모든 수치 정보를 구조화된 한국어로 판독해 주세요."
        if not self.selected_image_path:
            return

        selected_model = self.model_combo.currentText()

        self.btn_vision_run.setEnabled(False)
        self.vision_result_display.clear()
        self.vision_result_display.append(f"🧠 [로컬 비전 엔진] {selected_model} 멀티모달 가속 스캔 중...")

        # 이미지 Base64 인코딩 (메인 스레드에서 처리 - 빠름)
        try:
            with open(self.selected_image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            self.vision_result_display.append(f"❌ 이미지 읽기 오류: {str(e)}")
            self.btn_vision_run.setEnabled(True)
            return

        payload = {
            "model": selected_model,
            "prompt": question,
            "images": [base64_image],
            "stream": False
        }

        # 백그라운드 스레드 시작
        self.vision_thread = QThread()
        self.vision_worker = OllamaWorker(self.OLLAMA_URL, payload)
        self.vision_worker.moveToThread(self.vision_thread)

        self.vision_worker.progress.connect(self.update_vision_progress)
        self.vision_worker.finished.connect(self.on_vision_finished)
        self.vision_worker.error.connect(self.on_vision_error)

        self.vision_thread.started.connect(self.vision_worker.run)
        self.vision_thread.finished.connect(self.vision_thread.deleteLater)
        self.vision_thread.start()

    def update_vision_progress(self, msg):
        self.vision_result_display.append(msg)
        self.vision_result_display.ensureCursorVisible()

    def on_vision_finished(self, answer):
        self.vision_result_display.append(f"\n🏆 [로컬 멀티모달 시각 독해 성공]\n\n{answer}")
        self.btn_vision_run.setEnabled(True)
        if self.vision_thread and self.vision_thread.isRunning():
            self.vision_thread.quit()
            self.vision_thread.wait()
        self.vision_thread = None
        self.vision_worker = None

    def on_vision_error(self, error_msg):
        self.vision_result_display.append(f"\n❌ 오류: {error_msg}")
        self.btn_vision_run.setEnabled(True)
        if self.vision_thread and self.vision_thread.isRunning():
            self.vision_thread.quit()
            self.vision_thread.wait()
        self.vision_thread = None
        self.vision_worker = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    terminal = IfrsDualUltimateTerminal()
    terminal.show()
    sys.exit(app.exec_())
