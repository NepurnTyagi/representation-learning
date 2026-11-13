import torch
from torch import nn
import torchvision

class Dino(nn.Module):
    def __init__(self, Student, Teacher, img_dim, embed_dim, teacher_temperature, student_temperature, center_momentum, encoder_varient):
        super().__init__()
        self.Student = Student
        self.Teacher = Teacher
        self.Teacher.load_state_dict(self.Student.state_dict())
        for param in self.Teacher.parameters():
            param.requires_grad = False
        self.student_projection = nn.Linear(in_features=img_dim, out_features=embed_dim, bias=False)
        self.teacher_projection = nn.Linear(in_features=img_dim, out_features=embed_dim, bias=False)
        self.teacher_projection.load_state_dict(self.student_projection.state_dict())
        for param in self.teacher_projection.parameters():
            param.requires_grad = False
        self.embed_dim = embed_dim
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, embed_dim))
        self.encoder_varient = encoder_varient
        self.encode_image = self.encode_image_cls if self.encoder_varient=='CLS' else self.encode_image_gap
        
    def forward(self, X):
        X_global, X_local = X
        B = X_global.shape[0]//2
        X_student_global, X_student_local = self.student_projection(self.encode_image(self.Student, X_global)), self.student_projection(self.encode_image(self.Student, X_local))
        X_student_global, X_student_local = nn.functional.log_softmax(X_student_global/self.student_temperature, dim=-1), nn.functional.log_softmax(X_student_local/self.student_temperature,  dim=-1)
        with torch.no_grad():
            X_teacher = self.teacher_projection(self.encode_image(self.Teacher, X_global))
            batch_center = X_teacher.mean(dim=0)
            X_teacher = nn.functional.softmax((X_teacher-self.center)/self.teacher_temperature, dim=1)
            if self.training:
                self.center.mul_(self.center_momentum).add_(batch_center*(1-self.center_momentum))
        X_student_global, X_student_local = X_student_global.reshape(B, 2, self.embed_dim), X_student_local.reshape(B, 8, self.embed_dim)
        X_teacher = X_teacher.reshape(B, 2, self.embed_dim)
        loss = torch.einsum('btd,bsd->bts', X_teacher, X_student_global)
        mask = 1-torch.eye(loss.shape[1], device=loss.device)
        loss = (loss*mask.unsqueeze(0)).sum(dim=-1)
        loss += torch.einsum('btd,bsd->bts', X_teacher, X_student_local).sum(dim=-1)
        loss /= loss.shape[1]-1+X_student_local.shape[1]
        loss = loss.mean(dim=-1).mean(dim=-1)
        return -loss

    def encode_image_cls(self, encoder, X):
        X = encoder(X)
        X = X[:, 0, :]
        return X

    def encode_image_gap(self, encoder, X):
        X = encoder(X)
        X = X.mean(dim=1)
        return X

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_projection.eval()
        self.Teacher.eval()
        return self
