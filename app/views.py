from django.shortcuts import render
from django.views.generic import *
from .models import *
from django.contrib.auth.mixins import LoginRequiredMixin,UserPassesTestMixin
from django.conf import settings
from django.urls import reverse_lazy
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from django.shortcuts import render
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
# Create your views here.



class MyCreateClass(LoginRequiredMixin,CreateView):
    model= Cinema
    template_name='create.html'
    fields=['title','text','img','narx']
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

class UpdateView1(LoginRequiredMixin,UserPassesTestMixin,UpdateView):
    model = Cinema
    template_name = "update.html"
    fields=['title','text','img','narx']
    def test_func(self):
        obj = self.get_object()
        return obj.author == self.request.user

class DetailModel(DetailView):
    model=Cinema
    template_name='detail.html'

class Detail(DetailView):
    model = Cinema
    template_name='dtsh.html'

class Deleteview(LoginRequiredMixin,UserPassesTestMixin,DeleteView):
    model=Cinema
    template_name='delete.html'
    success_url = reverse_lazy('home')
    
    def test_func(self):
        obj = self.get_object()
        return obj.author == self.request.user
class Home(TemplateView):
    template_name="home.html"


def Mydef(request):
    if 'q' in request.GET:
        q=request.GET['q']
        cinema = Cinema.objects.filter(title__icontains=q) or Cinema.objects.filter(text__icontains=q) or Cinema.objects.filter(date__icontains=q)
    else:
        q=None
        cinema = Cinema.objects.all()
    return render(request, './news/news.html',{"cinema":cinema})



# GPT-2 model va tokenizerini yaratish
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def UZAI_Response(request):
    if request.method == 'POST':
        # Ma'lumotlarni tayyorlash
        text = request.POST.get('text')
        chat_messages = [{'sender': 'user', 'text': text}]

        # Tokenlar va attention_mask ni o'rganish
        input_ids = tokenizer.encode(text, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        # Modelga ma'lumotni o'tkazib, javobni olish
        with torch.no_grad():
            output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=50256, attention_mask=attention_mask)

        # Javobni chatga qo'shish
        response = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)  # input_ids[0] dan keyin kelgan qismi ajratib olish
        chat_messages.append({'sender': 'gpt2', 'text': response})

        return render(request, './chatUZ/UZ_AI.html', {'chat_messages': chat_messages})
    else:
        return render(request, './chatUZ/UZ_AI.html', {'chat_messages': []})
