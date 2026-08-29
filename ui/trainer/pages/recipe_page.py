"""Recipe creation and management page of the trainer."""

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.recipe_service import RecipeService

PACKAGE_FAMILIES = ["SO", "QFN", "QFP", "BGA", "DIP"]


class RecipePage(QWidget):
    """Lists recipes and edits their package metadata."""

    def __init__(self):
        super().__init__()

        self.recipe_service = RecipeService()
        self.selected_recipe = None
        self.parent_window = None

        self.setup_ui()
        self.load_recipe_list()

    def setup_ui(self):
        self.setWindowTitle("VisionAI Trainer")
        self.resize(1400, 850)

        self.setStyleSheet(
            """
            QWidget{
                background:#FFFFFF;
                font-family:'Poppins';
            }

            QLabel{
                color:#374151;
                font-size:12px;
            }

            QLineEdit,QComboBox,QTextEdit{
                border:1px solid #E5E7EB;
                border-radius:8px;
                padding:8px;
                background:#F9FAFB;
            }

            QListWidget{
                border:1px solid #E5E7EB;
                border-radius:10px;
                background:#F9FAFB;
            }

            QPushButton{
                background:#2563EB;
                color:white;
                border:none;
                border-radius:8px;
                padding:10px;
                font-weight:600;
            }

            QPushButton:hover{
                background:#1D4ED8;
            }
            """
        )

        root_layout = QHBoxLayout()

        # ----- Left panel: recipe list -----
        left_layout = QVBoxLayout()

        recipe_title = QLabel("Recipes")
        recipe_title.setStyleSheet("font-size:28px;font-weight:700;color:#000000;")

        self.recipe_list = QListWidget()
        self.recipe_list.itemClicked.connect(self.load_selected_recipe)

        self.btn_add_recipe = QPushButton("+ Add Recipe")
        self.btn_delete_recipe = QPushButton("Delete Recipe")
        self.btn_add_recipe.clicked.connect(self.new_recipe)
        self.btn_delete_recipe.clicked.connect(self.delete_recipe)

        left_layout.addWidget(recipe_title)
        left_layout.addWidget(self.recipe_list)
        left_layout.addWidget(self.btn_add_recipe)
        left_layout.addWidget(self.btn_delete_recipe)

        # ----- Right panel: recipe form -----
        right_layout = QVBoxLayout()

        header = QLabel("Recipe Configuration")
        header.setStyleSheet("font-size:28px;font-weight:700;color:#111827;")
        right_layout.addWidget(header)

        self.recipe_name = QLineEdit()

        self.package_family = QComboBox()
        self.package_family.addItems(PACKAGE_FAMILIES)

        self.package_type = QLineEdit()
        self.package_size = QLineEdit()
        self.package_version = QLineEdit()
        self.type_name = QLineEdit()
        self.pin_count = QLineEdit()
        self.notes = QTextEdit()

        form_fields = [
            ("Recipe Name", self.recipe_name),
            ("Package Family", self.package_family),
            ("Package Type", self.package_type),
            ("Package Size", self.package_size),
            ("Package Version", self.package_version),
            ("Type Name", self.type_name),
            ("Pin Count", self.pin_count),
        ]

        for label_text, widget in form_fields:
            right_layout.addWidget(QLabel(label_text))
            right_layout.addWidget(widget)

        right_layout.addWidget(QLabel("Notes"))
        right_layout.addWidget(self.notes)

        self.btn_save_recipe = QPushButton("Save Recipe")
        self.btn_save_recipe.clicked.connect(self.save_recipe)
        right_layout.addWidget(self.btn_save_recipe)

        root_layout.addLayout(left_layout, 1)
        root_layout.addLayout(right_layout, 3)

        self.setLayout(root_layout)

    # ------------------------------------------------------------------
    # Recipe list
    # ------------------------------------------------------------------

    def load_recipe_list(self):
        self.recipe_list.clear()
        self.recipe_list.addItems(self.recipe_service.get_recipe_list())

    def load_selected_recipe(self, item):
        recipe_name = item.text()
        recipe = self.recipe_service.load_recipe(recipe_name)
        if not recipe:
            return

        self.selected_recipe = recipe_name

        self.recipe_name.setText(recipe.get("recipe_name", ""))
        self.package_type.setText(recipe.get("package_type", ""))
        self.package_size.setText(recipe.get("package_size", ""))
        self.package_version.setText(recipe.get("package_version", ""))
        self.type_name.setText(recipe.get("type_name", ""))
        self.pin_count.setText(str(recipe.get("pin_count", 0)))
        self.notes.setPlainText(recipe.get("notes", ""))

        family = recipe.get("package_family", "SO")
        index = self.package_family.findText(family)
        if index >= 0:
            self.package_family.setCurrentIndex(index)

        if self.parent_window:
            self.parent_window.update_context(recipe)

        print(f"[RECIPE] Selected {recipe_name}")

    # ------------------------------------------------------------------
    # Create, save and delete
    # ------------------------------------------------------------------

    def new_recipe(self):
        """Clear the form for a new recipe."""
        self.selected_recipe = None

        self.recipe_name.clear()
        self.package_type.clear()
        self.package_size.clear()
        self.package_version.clear()
        self.type_name.clear()
        self.pin_count.clear()
        self.notes.clear()

    def save_recipe(self):
        recipe_name = self.recipe_name.text().strip()
        if not recipe_name:
            QMessageBox.warning(self, "Warning", "Recipe Name is required.")
            return

        recipe = self.recipe_service.get_recipe_template()
        recipe["recipe_name"] = recipe_name
        recipe["package_family"] = self.package_family.currentText()
        recipe["package_type"] = self.package_type.text()
        recipe["package_size"] = self.package_size.text()
        recipe["package_version"] = self.package_version.text()
        recipe["type_name"] = self.type_name.text()
        recipe["pin_count"] = int(self.pin_count.text() or 0)
        recipe["notes"] = self.notes.toPlainText()

        if self.recipe_service.recipe_exists(recipe_name):
            self.recipe_service.save_recipe(recipe_name, recipe)
        else:
            self.recipe_service.create_recipe(recipe_name, recipe)

        self.load_recipe_list()
        self.selected_recipe = recipe_name

        QMessageBox.information(self, "Success", "Recipe saved successfully.")

    def delete_recipe(self):
        if not self.selected_recipe:
            QMessageBox.warning(self, "Warning", "Please select a recipe.")
            return

        self.recipe_service.delete_recipe(self.selected_recipe)

        self.selected_recipe = None
        self.new_recipe()
        self.load_recipe_list()

        QMessageBox.information(self, "Success", "Recipe deleted.")
