import copy
# Định nghĩa class Buffer để lưu lịch sử selected_chs
class Buffer:
    def __init__(self, mem_size):
        """Khởi tạo buffer với kích thước cố định"""
        self.mem_size = mem_size
        self.history = []  # Danh sách lưu lịch sử selected_chs
    
    def add(self, selected_chs):
        """Thêm một danh sách selected_chs vào buffer"""
        # Tạo bản sao của selected_chs để tránh thay đổi ngoài ý muốn
        chs_copy = copy.deepcopy(selected_chs)
        self.history.append(chs_copy)
        # Nếu vượt quá kích thước, xóa phần tử cũ nhất
        if len(self.history) > self.mem_size:
            self.history.pop(0)
    
    def get_history(self):
        """Trả về toàn bộ lịch sử trong buffer"""
        return copy.deepcopy(self.history)
    
    def clear(self):
        """Xóa toàn bộ lịch sử trong buffer"""
        self.history = []
    
    def __len__(self):
        """Trả về số lượng phần tử trong buffer"""
        return len(self.history)
